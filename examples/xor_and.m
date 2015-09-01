%% Learning XOR and AND logical operations

% Configuration
neurons = [3 4 2];
a = 0.001;
bias = [-1 -1 -1];
x = [0 0;0 1;1 0; 1 1];
yref = [0 0;1 0;1 0;0 1];
lr = 0.7;
error = 10^-20;
maxIt = 100000;

% Processing
[W,E] = trainingMLP(neurons,a,bias,x,yref,lr,error,maxIt);

% Display Infos
disp('XOR AND')
disp('Input [0 0]');
outMLP(neurons,bias,[0 0],W)
disp('Input [0 1]');
outMLP(neurons,bias,[0 1],W)
disp('Input [1 0]');
outMLP(neurons,bias,[1 0],W)
disp('Input [1 1]');
outMLP(neurons,bias,[1 1],W)

% Plot MSE
semilogx(mse_hist)
