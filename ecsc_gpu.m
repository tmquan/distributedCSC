function res = ecsc_gpu(D0, S0, plan, isTrainingDictionary)
    %% If we want to train the dictionary
    if isempty(isTrainingDictionary)
        isTrainingDictionary = 1;
    end
    
    %% Parameters extractions
    elemSize = plan.elemSize;
    dataSize = plan.dataSize;
    atomSize = plan.atomSize;
    dictSize = plan.dictSize;
    blobSize = plan.blobSize;
    iterSize = plan.iterSize;
    numAtoms = blobSize(4);

	% plan.elemSize = [128, 128,  1,   1];
	% plan.dataSize = [128, 128,  1, 512]; % For example
	% plan.atomSize = [ 11,  11,  1,   1];
	% plan.dictSize = [ 11,  11,  1, 100];
	% plan.blobSize = [128, 128,  1, 100];
	% plan.iterSize = [128, 128,  1,  16];


    gNx = gpuArray(prod(iterSize));
    gNd = gpuArray(prod(iterSize));


    glambda = gpuArray(plan.lambda.Value);
    grho    = gpuArray(plan.rho.Value);
    gsigma  = gpuArray(plan.sigma.Value);

    %% Operators here
    %% Mean removal and normalisation projections
    Pzmn    = @(x) bsxfun(@minus,   x, mean(mean(mean(x,1),2),3));
    Pnrm    = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(sum(x.^2,1),2),3)));

    %% Projection of filter to full image size and its transpose
    % (zero-pad and crop respectively)
    Pzp     = @(x) zeropad(x, iterSize);
    PzpT    = @(x) bndcrop(x, dictSize);

    %% Projection of dictionary filters onto constraint set
    Pcn     = @(x) Pnrm(Pzp(Pzmn(PzpT(x))));

  	%% Memory reservation
    gS0     = gpuArray(S0);
    gD0     = gpuArray(D0);
    gD0     = Pnrm(gD0);

 	grx = gpuArray(Inf);
    gsx = gpuArray(Inf);
    grd = gpuArray(Inf);
    gsd = gpuArray(Inf);
    geprix = gpuArray(0);
    geduax = gpuArray(0);
    geprid = gpuArray(0);
    geduad = gpuArray(0);

  	gX      = gpuArray.zeros(iterSize);
    gY      = gpuArray.zeros(iterSize);
    gYprv   = gY;
    gXf     = gpuArray.zeros(iterSize);
    gYf     = gpuArray.zeros(iterSize);

    gS      = gS0; 
    %gSf     = gpuArray.zeros(dataSize);

    gD      = gpuArray.zeros(iterSize);
    gG      = gpuArray.zeros(iterSize);
    gGprv   = gpuArray.zeros(iterSize);

    gD      = gpuArray.zeros(iterSize);
    gG      = Pzp(gD); % Zero pad the dictionary
    gGprv   = gG;

    gDf     = gpuArray.zeros(iterSize);
    gGf     = gpuArray.zeros(iterSize);

    gU      = gpuArray.zeros(iterSize);
    gH      = gpuArray.zeros(iterSize);

    gGf     = gpuArray.zeros(iterSize);
    gGf     = fft3(gG);
    
    % Temporary buffers
    gGSf    = gpuArray.zeros(iterSize);
    gYSf    = gpuArray.zeros(iterSize);


    %% Set up algorithm parameters and initialise variables
    res  = struct('itstat', [], 'plan', plan);
    %% Main loops
    k = 1;
    tstart = tic;
    while k <= plan.MaxIter && (grx > geprix | gsx > geduax | ...
                                grd > geprid | gsd > geduad),
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Permutation here
        gS = gS0;
        %gS = permute(gS, [randperm(2), 3, 4]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        k = k+1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end %% End main loop

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Collect the output
    res.D = gather(gD0);
    %res = [];
    
end