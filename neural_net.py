import numpy as np

class activations:
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def relu(z):
        return np.maximum(0,z)
    
    @staticmethod
    def softmax(z):
        return np.exp(z)/np.sum(np.exp(z),axis=0)
    
architecture_dict = {
            'name': 'default_net',
            'input_dim': (None, 10),
            'n_layers': 1,
            'layer_dims': [1],
            'layer_activations': [activations.sigmoid]
        }

hyperparam_dict = {
            'n_epoch': 10,
            'learning_rate': 0.01,
            'batch_size': 100
        }

class nn:
    
    def __init__(self, arch_dict = architecture_dict, hyper_dict = hyperparam_dict):
        self.arch_dict = arch_dict
        self.hyper_dict = hyper_dict
        self.__verbose = {
                'hiphen': '------------',
                'nothing': '',
                'tilde': '~~~~'
                }
    
    def set_architecture(self, architecture):
        self.arch_dict = architecture
    
    def set_hyperparams(self, hyper_dict):
        self.hyper_dict = hyper_dict
    
    def derivative(self, func, x, h=1e-10):
        return (func(x+h) - func(x))/h
        
    def __verbose_print(self, x, verbose, side = 'hiphen'):
        if(verbose):
            print(self.__verbose[side],x,self.__verbose[side])
        
    def __init_weights(self):
        self.__weight_dict = {}
        
        n_layers = self.arch_dict['n_layers']
        layers = self.arch_dict['layer_dims']
        input_dim = self.arch_dict['input_dim']
        
        prev_layer =  input_dim[-1]
        for l in range(n_layers):
            W = np.random.randn(layers[l],prev_layer)
            b = np.zeros((layers[l],1))
            prev_layer = layers[l]
            self.__weight_dict['W_'+str(l+1)] = W
            self.__weight_dict['b_'+str(l+1)] = b
            
    def __build_net(self, X, Y,verbose):
        self.__verbose_print('Building Net', verbose)
        
        self.arch_dict['input_dim'] = np.shape(X)
        self.__init_weights()
        
    def __forward_prop(self, X):
        n_layers = self.arch_dict['n_layers']
        
        self.__forward_dict = {}
        
        A_prev = X
        self.__forward_dict['A_0'] = X
        for l in range(n_layers):
            W = self.__weight_dict['W_'+str(l+1)]
            b = self.__weight_dict['b_'+str(l+1)]
            activation_fn = self.arch_dict['layer_activations'][l]
            
            Z = np.dot(W,A_prev) + b
            A = activation_fn(Z)
            A_prev = A
            
            self.__forward_dict['Z_'+str(l+1)] = Z
            self.__forward_dict['A_'+str(l+1)] = A
           
        return A_prev
    
    def __compute_cost(self, A, Y):
        m = np.shape(A)[-1]
        
        activation_fn = self.arch_dict['layer_activations'][-1]
        
        if activation_fn is activations.sigmoid:
            J = (-1/m) * np.sum(
                    np.multiply(Y, np.log(A)) 
                    + 
                    np.multiply((1-Y),np.log((1-A)))
                    )
            
        elif activation_fn is activations.softmax:
            J = (-1/m) * np.sum(np.multiply(Y, np.log(A)))
        
        return J
    
    def __backward_prop(self, X, Y):
        n_layers = self.arch_dict['n_layers']
        m = X.shape[1]
        activation_fn = self.arch_dict['layer_activations'][-1]
        
        self.__backward_dict = {}
        
            
        A = self.__forward_dict['A_'+str(n_layers)]
        dA = -Y/A + (1-Y)/(1-A)
        dZ = A - Y
        self.__backward_dict['dA_'+str(n_layers)] = dA
        self.__backward_dict['dZ_'+str(n_layers)] = dZ
        for l in range(n_layers-1, -1, -1):
            dA = self.__backward_dict['dA_'+str(l+1)]
            Z = self.__forward_dict['Z_'+str(l+1)]
            W = self.__weight_dict['W_'+str(l+1)]
            A_prev = self.__forward_dict['A_'+str(l)]
            
            activation_fn = self.arch_dict['layer_activations'][l]
            
            if(l==n_layers-1):
                dZ = self.__backward_dict['dZ_'+str(n_layers)]
            else:
                dAZ = self.derivative(activation_fn, Z)
                dZ = dA * dAZ
            dW = np.dot(dZ,A_prev.T) / m   
            db = np.sum(dZ, axis=1, keepdims=True) / m 
            dA_prev = np.dot(W.T,dZ)            
            
            self.__backward_dict['dA_'+str(l)] = dA_prev
            self.__backward_dict['dZ_'+str(l+1)] = dZ
            self.__backward_dict['dW_'+str(l+1)] = dW
            self.__backward_dict['db_'+str(l+1)] = db
            
            
    def __update_weights(self):
        alpha = self.hyper_dict['learning_rate']
        n_layers = self.arch_dict['n_layers']
        
        for l in range(n_layers):
            W = self.__weight_dict['W_'+str(l+1)]
            b = self.__weight_dict['b_'+str(l+1)]
            dW = self.__backward_dict['dW_'+str(l+1)]
            db = self.__backward_dict['db_'+str(l+1)]
            
            W = W - (alpha * dW)
            b = b - (alpha * db)
            
            self.__weight_dict['W_'+str(l+1)] = W
            self.__weight_dict['b_'+str(l+1)] = b  
        
    def fit(self, X, Y, verbose=True):
        self.__result_dict = {}
        self.__build_net(X, Y,verbose)
        
        epoch = self.hyper_dict['n_epoch']
        batch_size = self.hyper_dict['batch_size']
        n_examples = self.arch_dict['input_dim'][0]
        
        self.__verbose_print('Training Net',verbose)
        
        for e in range(epoch):
            self.__verbose_print('Epoch - '+str(e+1)+'/'+str(epoch), verbose, side = 'nothing')
            cost = 0
            for b in range(0, n_examples - batch_size, batch_size):
                self.__verbose_print(
                        'Batch - '+str(b//batch_size + 1)+':'+str(b+1)+'-'+str(b+batch_size), 
                        verbose, 
                        side = 'tilde')
                X_batch = X[b:b+batch_size].T
                Y_batch = Y[b:b+batch_size].T
                A = self.__forward_prop(X_batch)
                cost += self.__compute_cost(A, Y_batch)
                self.__backward_prop(X_batch, Y_batch)
                self.__update_weights()
            self.__result_dict['cost_'+str(e+1)] = cost
            self.__verbose_print('Cost - '+ str(cost), verbose, side = 'nothing')
            
        return self.get_costs()
    
    def get_costs(self):
        costs = []
        epochs = self.hyper_dict['n_epoch']
        
        for i in range(epochs):
            costs.append(self.__result_dict['cost_'+str(i+1)])
        return costs
    
    def predict(self, X, batch_size = 100, verbose = True):
        n_examples = np.shape(X)[0]
        predictions = np.zeros((X.shape[0],self.arch_dict['layer_dims'][-1]))
        for b in range(0, n_examples - batch_size, batch_size):
            self.__verbose_print(
                        'Batch - '+str(b//batch_size + 1)+':'+str(b+1)+'-'+str(b+batch_size), 
                        verbose, 
                        side = 'tilde')
            X_batch = X[b:b+batch_size].T
            A = self.__forward_prop(X_batch).T
            predictions[b:b+batch_size] = A
        
        return predictions