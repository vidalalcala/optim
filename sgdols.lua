----------------------------------------------------------------------
--[[ A plain implementation of SGD with OLS estimate of the Hessian

   ARGS:
-  opfunc : a function that takes a single input (X), the point of 
            evaluation, and returns f(X) and df/dX
-  x      : the initial point
-  config : a table with configuration parameters for the optimizer
-  config.learningRate	   : learning rate
-  config.gamma		   : learning power law		   
-  state  : a table describing the state of the optimizer; after each
            call the state is modified
-  state.evalCounter	   : number of stochastic optimization samples
-  state.numParameters	   : parameter dimension
-  state.P                 : (p + 1) x ( p + 1 ) matrix
-  state.B                 : p x (p+1) matrix (OLS beta)
-  state.G                 : p x p matrix (OLS inverse beta)
-  state.Gt                : state.G transpose
-  state.parametersSlow	   : parameter without Polyak averaging

   RETURN:
-  x     : the new x vector
-  f(x)  : the function, evaluated before the update

(Jose V. Alcala-Burgos, 2013)
]]

function optim.sgdols(opfunc, x, config, state)
   
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1.0
   local gamma = config.gamma or 0.60
   
   state.evalCounter = state.evalCounter or 0
   state.numParameters = state.numParameters or x:size(1)
   
   local nevals = state.evalCounter
   local p = state.numParameters
   
   if torch.getdefaulttensortype() == 'torch.CudaTensor' then
      torch.setdefaulttensortype('torch.FloatTensor')
      
      state.P = state.P or torch.eye(p + 1):cuda()
      state.B = state.B or torch.Tensor(p + 1, p):zero():cuda()
      state.G = state.G or torch.eye(p):cuda()
      state.Gt = state.Gt or torch.eye(p):cuda()
   
      torch.setdefaulttensortype('torch.CudaTensor') 
   else
      state.P = state.P or torch.eye(p + 1)
      state.B = state.B or torch.Tensor(p + 1, p):zero()
      state.G = state.G or torch.eye(p)
      state.Gt = state.Gt or torch.eye(p) 
   end
   
   state.parametersSlow = state.parametersSlow or x:clone()
   
   local P = state.P
   local B = state.B
   local G = state.G
   local Gt = state.Gt
   
   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(state.parametersSlow)
   
   -- (2) update evaluation counter
   state.evalCounter = state.evalCounter + 1
   
   -- (3) learning rate decay (annealing)
   local clr = lr / ( (1.0 + nevals)^(gamma) )
   
   
   -- (5) save old parameter
   local xOne = torch.Tensor(p + 1)
   local xRest = xOne:narrow(1, 1, p)
   local y = torch.Tensor(p)
   y[{}] = dfdx
   xRest[{}] = state.parametersSlow
   xOne[p + 1] = 1.0
   
   state.parametersSlow:addmv( -clr/2.0 , G , y )
   state.parametersSlow:addmv( -clr/2.0 , Gt , y )
   
   x:mul( (state.evalCounter -1)/state.evalCounter )
   x:add( state.evalCounter , state.parametersSlow )
   
   -- (7) rank one update of matrices
   local Px = torch.Tensor(p + 1)
   Px:zero()
   Px:addmv(P ,xOne)
   local a = xOne:dot(Px)
   a = a + 1.0
   local alpha = 1.0/a
   local u = torch.Tensor(p + 1, 1)
   u = Px:narrow(1, 1, p)
   u:mul(alpha)
   local v = torch.Tensor(p)
   v[{}] = y
   v:addmv(- 1.0, B:t(), xOne )
   state.B:addr(alpha, Px, v)
   state.P:addr(-alpha, Px, Px)
   
   local Gu = torch.Tensor(p)
   Gu:zero()
   Gu:addmv(G, u)
   local Gv = torch.Tensor(p)
   Gv:zero()
   Gv:addmv(G:t(), v)
   local b = v:dot(Gu)
   b = b + 1.0
   local beta =  1.0/b
   
   state.G:addr(-beta, Gu, Gv )
   state.Gt:addr(-beta, Gv, Gu )
   
   -- return x*, f(x) before optimization
   return x,{fx}
end

