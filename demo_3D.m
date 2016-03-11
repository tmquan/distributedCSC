clc; clear all; close all;

%% 
addpath(genpath('.'));
g = gpuDevice(1);
reset(g)

%load kiwi_128_uint8
% vol = imreadtif('cthead128.tif');
% vol = imreadtif('kiwi128.tif');% vol = permute(vol, [1 3 2]);
% vol = imreadtif('pomegranate128.tif');
% vol = imreadtif('bonsai.tif');
% vol = imreadtif('foot.tif');
vol = imreadtif('zebrafish.tif');

S0 = vol;
S0 = im2single(S0);

% S0 = imreadtif('em.tif');
% S0 = single(S0);
% S0 = S0(:,:,1:1);
% S0 = scale1(S0);

% S0 = S0-mean(vec(S0));
% [Sl, Sh] = lowpass(S0, 0.1, 5);
% S0 = Sh;
S0 = scale1(S0);
%% Seed the randomness
rng(2016);

plan.elemSize = [128, 128,  128,  1];
plan.dataSize = [128, 128,  128,  1]; % For example
plan.atomSize = [ 11,  11,  11,   1];
plan.dictSize = [ 11,  11,  11,  32];
plan.blobSize = [128, 128,  128, 32];
plan.iterSize = [128, 128,  128, 32]; 

% plan.elemSize = [256, 256,  256,  1];
% plan.dataSize = [256, 256,  256,  1]; % For example
% plan.atomSize = [ 11,  11,  11,   1];
% plan.dictSize = [ 11,  11,  11,  16];
% plan.blobSize = [256, 256,  256, 16];
% plan.iterSize = [256, 256,  256, 16]; 

%% Initialize the plan
plan.alpha  = params; % See param.m
plan.gamma  = params; 
plan.delta  = params;
plan.theta  = params;
plan.omega  = params;
plan.lambda = params; 
plan.sigma  = params; 
plan.rho    = params; 

% plan.lambda.Value	= .01; %10; 1; 0.1; 0.01; 0.001; 
% plan.weight         = 100;
% plan.sigma.Value	= .5;
% plan.rho.Value		= .5;
% plan.sigma.AutoScaling 	= 1;
% plan.rho.AutoScaling 	= 1;
plan.lambda.Value	= 0.01; %10; 1; 0.1; 0.01; 0.001; 
plan.weight         	= 10;
plan.sigma.Value		= 10;
plan.rho.Value		= 10;
plan.sigma.AutoScaling 	= 0;
plan.rho.AutoScaling 	= 0;

%% Solver initialization
plan.Verbose = 1;
plan.MaxIter = 500;
plan.AbsStopTol = 1e-6;
plan.RelStopTol = 1e-6;

%% Initialize the dictionary
D0 = zeros(plan.dictSize, 'single'); % Turn on if using single precision
D0 = rand(plan.dictSize);
size(S0)
plan.dataSize
S0 = reshape(S0, plan.dataSize);

%% Run the CSC algorithm
isTrainingDictionary=1;
[resD] = ecsc_gpu(D0, S0, plan, isTrainingDictionary);

%%
close all;
%plan.lambda.Value	= 0.01;
% plan.dataSize = [128, 128,  1, 1];
[resX] = ecsc_gpu(resD.G, S0, plan, 0);
% Slicer(squeeze(resX.Y));
% Slicer(squeeze(resX.GY));

%figure; imagesc(squeeze(sum(resX.GY, 4))); axis equal off; colormap gray; drawnow;
%%
% [s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'cthead128_', 'maps_cthead/');
% [s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'kiwi128_', 'maps_kiwi/');
% [s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'pomegranate128_', 'maps_pomegranate/');
% [s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'bonsai_', 'maps_bonsai/');
% [s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'foot_', 'maps_foot/');
[s, d, y, gy, gs] = saveMaps(S0, resX.G, resX.Y, plan, 'zebrafish_', 'maps_zebrafish/');