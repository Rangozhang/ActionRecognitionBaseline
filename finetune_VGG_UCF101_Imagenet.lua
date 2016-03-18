require 'nn'
require 'cutorch'
require 'loadcaffe'
require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'optim'
local nninit = require 'nninit'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	GPU=4, 
	nGPU=1,
	backend='cudnn', 
	epoch=20, 
	batchSize=64,
	epochSize=5000,
    nTestBatch=25,
	manualSeed=2,
	save='checkpoint',
	datapath='../dataset/UCF101/videos',
	listpath='../dataset/UCF101/ucfTrainTestlist',
	nInputFrames=1,
	learningRate = 1e-3   -- for finetune layers, *0.01 for unfinetune layers
}

paths.dofile('util.lua')

print(opt)

--1. load from caffe
-- --[[
model = loadcaffe.load('./pretrained/VGG_ILSVRC_16_layers_deploy.prototxt', './pretrained/VGG_ILSVRC_16_layers.caffemodel') -- , opt.backend)
-- remove the last softmax layer, otherwise would error on 2# parameter nil
model:remove(40)
model:remove(39)
model:add(nn.Linear(4096, 101):init('weight',nninit.addNormal, 0, 0.1))
model:add(nn.LogSoftMax())
cudnn.convert(model, cudnn)

-- 2. load from original model
torch.save('model.t7', model)
--]]

cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)

-- model = loadDataParallel('./checkpoint/model_5_good.t7')
-- model = torch.load('model.t7')


function parallelFeatureLayers(model)
    model:cuda()
    features = nn.Sequential()
    for i = 1,31 do
	features:add(model:get(1))
	model:remove(1)
    end
    features:cuda()
    features = makeDataParallel(features, opt.nGPU)
    local newModel = nn.Sequential()
    newModel:add(features):add(model)
    newModel:cuda()
    return newModel
end

collectgarbage()

print("Parallelrize the model...")
model = parallelFeatureLayers(model)

print(model)
-- io.read()
local parameters, gradParameters = model:getParameters()

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p '..opt.save)

-- change learning rate and change the model's last layer to logSoftMax
lrs_model = model:clone()
lrs = lrs_model:getParameters()

function finetune(lr)
    lrs:fill(lr*1e-2)
    tmp_model = lrs_model:get(2)
    for i = 1,9 do
        if tmp_model:get(i).weight ~= nil then
    	    tmp_model:get(i).weight:fill(lr)
    	    tmp_model:get(i).bias:fill(lr)
        end
        if i == 8 then
	    tmp_model:get(i).weight:fill(lr*10)
	    tmp_model:get(i).bias:fill(lr*10)
        end
    end
end

print("Set learning rate for layers")
finetune(opt.learningRate)

local optimState = {
	learningRates = lrs,
	learningRateDecay = 1e-6,
	momentum = 0.9,
	dampening = 0.0,
	weightDecay = 0.0
}

print(optimState)

collectgarbage()

----------------------------data loading---------------------------
-- --[[
package.path = "../?.lua;" .. package.path
require 'datasources.ucf101'
require 'datasources.thread'

--[[
-- test
package.path = "../?.lua;" .. package.path
require 'datasources.ucf101'
require 'datasources.augment'
dataLoader = AugmentDatasource(UCF101Datasource(opt), {resize={224, 224}, rgb_mean={123.68, 116.779, 103.939}, rgb2bgr=true})
dataLoader:cuda()
--]]

local option = opt
dataLoader = ThreadedDatasource(
    function()
	local opt = option
	package.path = "../?.lua;" .. package.path
	require 'datasources.ucf101'
	require 'datasources.augment'
	return AugmentDatasource(UCF101Datasource(opt), {resize={224, 224}, mean={123.68, 116.779, 103.939}, rgb2bgr=true})
    end, {nDonkey=2})
dataLoader:cuda()
print('Data loading initialization finished')

--[[
timer = torch.Timer()
for i = 1, 10000 do
    batch, label = dataLoader:nextBatch(opt.batchSize, 'train')
    print{batch}
    print{label}
end
print(timer:time())
io.read()
--]]
----------------------------training function-----------------------
criterion = nn.ClassNLLCriterion()
criterion:cuda()

epoch = 1
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

function train()
   print('==> doing epoch on training data:')
   print('==> online epoch # ' .. epoch)

   if epoch == 5 then
   	finetune(opt.learningRate * 1e-2)   
	opt.learningRateDecay = 1e-6
   elseif epoch == 10 then
	finetune(opt.learningRate * 1e-4)
        opt.learningRateDecay = 0.0
   elseif epoch == 15 then
	finetune(opt.learningRate * 1e-6)
	opt.learningRateDecay = 0.0
   end

   batchNumber = 0
   cutorch.synchronize()
   model:training()
   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
	local batch, label = dataLoader:nextBatch(opt.batchSize, 'train')	
	batch = torch.squeeze(batch[{{},{1}}])
	trainBatch(batch, label)
   end

   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   local function sanitize(net)
      local list = net:listModules()
      for _,val in ipairs(list) do
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if (name == 'output' or name == 'gradInput') then
                  val[name] = field.new()
               end
            end
      end
   end
   sanitize(model)

   -- this function helps to get rid of the parallel table
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   epoch = epoch + 1
end

local timer = torch.Timer()
local dataTimer = torch.Timer()
-- local inputs = torch.CudaTensor()
-- local labels = torch.CudaTensor()

function trainBatch(inputs, labels) -- (inputsCPU, labelsCPU)
    cutorch.synchronize()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()
    collectgarbage()

    -- inputs:resize(inputsCPU:size()):copy(inputsCPU)
    -- labels:resize(labelsCPU:size()):copy(labelsCPU)

    local loss, outputs

    feval = function(x)
	    if x ~= parameters then parameters:copy(x) end
        --[[
        print('in feval: ')
        print('parameter sum: ')
	    print(parameters:sum())
	    print('inputs sum: ')
        print(inputs:sum())
	    print('outputs sum: ')
	    print(outputs:sum())
	    --]]
        outputs = model:forward(inputs)
	    model:zeroGradParameters()
	    loss = criterion:forward(outputs, labels)
	    local gradOutputs = criterion:backward(outputs, labels)
	    model:backward(inputs, gradOutputs)
        -- print('gradParameters sum: ')
	    -- print(gradParameters:sum())
	    return loss, gradParameters
    end
    
    optim.sgd(feval, parameters, optimState)

    model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss_epoch = loss_epoch + loss
    -- top-1 error
    local top1 = 0
    do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      -- --[[
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labels[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      --]]
      -- print('top1: ' .. top1)
      top1 = top1 * 100 / opt.batchSize;
    end
    -- Calculate top-1 error, and print information
    print(('Epoch: [%d][%d/%d]\tTime %.3f Loss %.4f Top1-%%: %.2f LR TBA DataLoadingTime %.3f'):format(epoch, batchNumber, opt.epochSize, timer:time().real, loss, top1, dataLoadingTime))
    dataTimer:reset()
end

----------------------------test function---------------------------
local testTimer = torch.Timer()
local top1_center, Testloss
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
tmp_batch = torch.CudaTensor()
tmp_label = torch.CudaTensor()
function test()
    print('==> doing epoch on validation data:')
    print("==> online epoch # " .. epoch)

    batchNumber = 0
    cutorch.synchronize()
    testTimer:reset()

    model:evaluate()
    
    top1_center = 0
    Testloss = 0

    i = 0
    for batch, label in dataLoader:orderedIterator(16, 'test') do
        if batch:nDimension() ~= 0 then 
            i = i+1
            batch = torch.squeeze(batch,2)
            testBatch(batch, label)
            if i == opt.nTestBatch then
                break
            end
        end
    end

    cutorch.synchronize()

    top1_center = top1_center*100 / opt.nTestBatch
    Testloss = Testloss / (opt.nTestBatch)
    testLogger:add{
        ['% top1 accuracy (test set) (center crop)'] = top1_center,
        ['avg loss (test set)'] = Testloss
    }
    print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                            .. 'average loss (per batch): %.2f \t '
                            .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                            epoch, timer:time().real, Testloss, top1_center))   
    print('\n')
end

function testBatch(inputs, labels)
    batchNumber = batchNumber + 1   
    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, labels)
    cutorch.synchronize()
    local pred = outputs:float()

    Testloss = Testloss + err

    local _, pred_sorted = pred:sort(2, true)
    for i=1,pred:size(1) do
        local g = labels[i]
        if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
    end
end

----------------------------actual training-------------------------
for i = 1,opt.epoch do
    train()
    test()
end
