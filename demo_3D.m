clc; clear all; close all;

%% 
addpath(genpath('.'));
gpuDevice(1);

%% Load the image
img = imread('lena.png');
img = rgb2gray(img);

S0 = img;
S0 = single(S0);
S0 = scale1(S0);

%% Seed the randomness
rng(2016);

plan.elemSize = [128, 128,  1,   1];
plan.dataSize = [128, 128,  1, 512]; % For example
plan.atomSize = [ 11,  11,  1,   1];
plan.dictSize = [ 11,  11,  1, 100];
plan.blobSize = [128, 128,  1, 100];
plan.iterSize = [128, 128,  1,  16];


%% Initialize the plan
plan.alpha  = params; % See param.m
plan.gamma  = params;
plan.delta  = params;
plan.theta  = params;
plan.omega  = params;
plan.lambda = params; 
plan.sigma  = params; 
plan.rho    = params; 

plan.lambda.Value	= 0.5;
plan.weight         = 0.01;
plan.sigma.Value	= 0.5;
plan.rho.Value		= 0.5;

%% Solver initialization
plan.Verbose = 1;
plan.MaxIter = 500;
plan.AbsStopTol = 1e-6;
plan.RelStopTol = 1e-6;

%% Initialize the dictionary
%D0 = zeros(plan.dictSize); % Turn on if using single precision
D0 = rand(plan.dictSize);

%% Run the CSC algorithm
isTrainingDictionary=1;
[res] = ecsc_gpu(D0, S0, plan, isTrainingDictionary);


