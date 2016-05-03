require('nn')
require('nngraph')
require('base')
stringx = require('pl.stringx')
require 'io'
ptb = require('data')

local params = {
                batch_size=1, -- minibatch
                seq_length=1, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0.3, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=5,  -- when to start decaying learning rate
                max_max_epoch=11, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }

model = {}

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function run_test(words, len)
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    new_sent = ""
    for i = 1, words:size(1) do
        local x = torch.Tensor(1):fill(words[i])
        tmp_y, model.s[1] = unpack(model.rnns[1]:forward({x, model.s[0]}))
        tmp_y = torch.exp(tmp_y)
        pred = torch.multinomial(tmp_y, 1)
        g_replace_table(model.s[0], model.s[1])
        if i > 1 then
          new_sent = new_sent .. " "
        end
        new_sent = new_sent .. dict[x[1]]
    end

    local x = torch.Tensor(1):fill(words[words:size(1)])
    for i = 1, len do
        tmp_y, model.s[1] = unpack(model.rnns[1]:forward({x, model.s[0]}))
        tmp_y = torch.exp(tmp_y)
        pred = torch.multinomial(tmp_y, 1)
        g_replace_table(model.s[0], model.s[1])
        x[1] = pred[1][1]
        new_sent = new_sent .. " " .. dict[pred[1][1]]
    end
    print(new_sent)
    g_enable_dropout(model.rnns)
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  x = torch.zeros(#line - 1)
  for i = 2,#line do
    if vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
    x[i - 1] = vocab_map[line[i]]
  end
  return tonumber(line[1]), x, line
end

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

function create_network()
    local x                  = nn.Identity()()
    local prev_s             = nn.Identity()()
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
    local next_s             = {}
    local split              = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped))
    --local err                = nn.ClassNLLCriterion()({pred})
    local module             = nn.gModule({x, prev_s},
                                      {pred, nn.Identity()(next_s)})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module
end

function setup()
    print("Creating a RNN LSTM network.")
    local core_network = create_network()
    core_network:getParameters().copy(old_model.core_network:getParameters())
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = torch.zeros(params.batch_size, params.rnn_size)
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = torch.zeros(params.batch_size, params.rnn_size)
        model.ds[d] = torch.zeros(params.batch_size, params.rnn_size)
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = torch.zeros(params.seq_length)
end

function construct_dict()
    dict = {}
    for key, value in pairs(ptb.vocab_map) do
        dict[value] = key
    end
    return dict
end

old_model = torch.load('lstm.model')
state_train = {data=ptb.traindataset(params.batch_size)}
state_valid =  {data=ptb.validdataset(params.batch_size)}
state_test =  {data=ptb.testdataset(params.batch_size)}
setup()
vocab_map = ptb.vocab_map
dict = construct_dict()
words = {}

while true do
  print("Query: len word1 word2 etc")
  local ok, len, x, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    run_test(x, len)
    io.write('\n')
  end
end
