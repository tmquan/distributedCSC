clc; clear all; close all;

%% 
addpath(genpath('.'));
g = gpuDevice(2);
reset(g)

load kiwi_128_uint8
S0 = vol;
% S0 = imread('kiwi_128.png');
% S0 = imread('brain_128.png');
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


%% Initialize the plan
plan.alpha  = params; % See param.m
plan.gamma  = params; 
plan.delta  = params;
plan.theta  = params;
plan.omega  = params;
plan.lambda = params; 
plan.sigma  = params; 
plan.rho    = params; 

plan.lambda.Value	= .01; %10; 1; 0.1; 0.01; 0.001; 
plan.weight         = .1;
plan.sigma.Value	= .05;
plan.rho.Value		= .05;
plan.sigma.AutoScaling 	= 1;
plan.rho.AutoScaling 	= 1;

%% Solver initialization
plan.Verbose = 1;
plan.MaxIter = 1000;
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
plan.dataSize = [128, 128,  1, 1];
[resX] = ecsc_gpu(resD.G, S0(:,:,:,1), plan, 0);
Slicer(squeeze(resX.Y));
Slicer(squeeze(resX.GY));

%figure; imagesc(squeeze(sum(resX.GY, 4))); axis equal off; colormap gray; drawnow;
%%
% [s, d, y, gy, gs] = saveMaps(S0(:,:,:,1), resX.G, resX.Y, plan, 'em_single_128_', 'maps_em_single/');
% [s, d, y, gy, gs] = saveMaps(S0(:,:,:,1), resX.G, resX.Y, plan, 'em_multi_128_', 'maps_em_multi/');
[s, d, y, gy, gs] = saveMaps(S0(:,:,:,1), resX.G, resX.Y, plan, 'brain_128_', 'maps_brain/');
% [s, d, y, gy, gs] = saveMaps(S0(:,:,:,1), resX.G, resX.Y, plan, 'kiwi_128_', 'maps_kiwi/');
% [s, d, y, gy, gs] = saveMaps(S0(:,:,:,1), resX.G, resX.Y, plan, 'lena_128_', 'maps_lena/');