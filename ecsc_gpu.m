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
    numIters = numAtoms/iterSize(4);
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
    PzpT    = @(x) bndcrop(x, [dictSize(1:end-1), iterSize(4)]);

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
        %% Compute the signal in DFT domain
        gSf  = fft3(gS); 

        %% Extract the atom iteration
        for iter = 1:numIters
        	chunk = 1:iterSize(4);
        	march = chunk+(iter-1)*iterSize(4); % marching through the dictionary
        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        	gD 		= gD0(:,:,:,march);
        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        	gG      = Pzp(gD); % Zero pad the dictionary, PARTIALLY
        	gGf  	= fft3(gG);
        	% size(gGf)
        	% size(gSf)
        	gGSf 	= bsxfun(@times, conj(gGf), gSf);

        	%% Solve X subproblem
        	gXf  = solvedbi_sm(gGf, grho, gGSf + grho*fft3(gY-gU)); 
        	gX   = ifft3(gXf); 
        	gXr  = gX; %relaxation

        	%% Solve Y subproblem
        	gY   = shrink(gXr + gU, (glambda/grho)*plan.weight); % Adjust threshold 
        	gYf  = fft3(gY);
        	gYSf = sum(bsxfun(@times, conj(gYf), gSf), 5);

        	%% Solve U subproblem
        	gU = gU + gXr - gY;
        	
        	%% Update params 
        	gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
        	grx = norm(vec(gX - gY))/max(gnX,gnY);
        	gsx = norm(vec(gYprv - gY))/gnU;
        	geprix = sqrt(gNx)*plan.AbsStopTol/max(gnX,gnY)+plan.RelStopTol;
        	geduax = sqrt(gNx)*plan.AbsStopTol/(grho*gnU)+plan.RelStopTol;

        	if plan.rho.Auto,
	            if k ~= 1 && mod(k, plan.rho.AutoPeriod) == 0,
	                if plan.rho.AutoScaling,
	                    grhomlt = sqrt(grx/gsx);
	                    if grhomlt < 1, grhomlt = 1/grhomlt; end
	                    if grhomlt > plan.rho.Scaling, grhomlt = gpuArray(plan.rho.Scaling); end
	                else
	                    grhomlt = gpuArray(plan.rho.Scaling);
	                end
	                grsf = 1;
	                if grx > plan.rho.RsdlRatio*gsx, grsf = grhomlt; end
	                if gsx > plan.rho.RsdlRatio*grx, grsf = 1/grhomlt; end
	                grho = grsf*grho;
	                gU = gU/grsf;
	            end
	        end

	        %% Record information
	        gYprv = gY;

	        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        	%% Solve D subproblem
        	%size(gYSf)
        	gDf  = solvedbi_sm(gYf, gsigma, gYSf + gsigma*fft3(gG - gH));
        	gD   = ifft3(gDf);
        	gDr  = gD;

        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        	%% Solve G subproblem
        	gG   = Pcn(gDr + gH);
        
        	%% Solve H subproblem
        	gH = gH + gDr - gG;

        	%% Update params    
	        gnD = norm(gD(:)); gnG = norm(gG(:)); gnH = norm(gH(:));
	        grd = norm(vec(gD - gG))/max(gnD,gnG);
	        gsd = norm(vec(gGprv - gG))/gnH;
	        geprid = sqrt(gNd)*plan.AbsStopTol/max(gnD,gnG)+plan.RelStopTol;
	        geduad = sqrt(gNd)*plan.AbsStopTol/(gsigma*gnH)+plan.RelStopTol;
	        
	        if plan.sigma.Auto,
	            if k ~= 1 && mod(k, plan.sigma.AutoPeriod) == 0,
	                if plan.sigma.AutoScaling,
	                    gsigmlt = sqrt(grd/gsd);
	                    if gsigmlt < 1, gsigmlt = 1/gsigmlt; end
	                    if gsigmlt > plan.sigma.Scaling, gsigmlt = gpuArray(plan.sigma.Scaling); end
	                else
	                    gsigmlt = gpuArray(plan.sigma.Scaling);
	                end
	                gssf = gpuArray(1);
	                if grd > plan.sigma.RsdlRatio*gsd, gssf = gsigmlt; end
	                if gsd > plan.sigma.RsdlRatio*grd, gssf = 1/gsigmlt; end
	                gsigma = gssf*gsigma;
	                gH = gH/gssf;
	            end
	        end
	        %% Record information
	        gGprv = gG;
        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	        %% Collect information
	        % Compute l1 norm of Y
	        gJl1 = sum(abs(vec( gY)));
	        % Compute measure of D constraint violation
	        gJcn = norm(vec(Pcn(gD) - gD));
	        % Compute data fidelity term in Fourier domain (note normalisation)
	        gJdf = sum(vec(abs(sum(bsxfun(@times,gGf,gYf),4)-gSf).^2))/(2*prod(blobSize));
	        gJfn = gJdf + glambda*gJl1;
	        % Record and display iteration details
	        tk = toc(tstart);
	        res.itstat = [res.itstat;...
	            [k gather(gJfn) gather(gJdf) gather(gJl1) gather(grx) gather(gsx)...
	            gather(grd) gather(gsd) gather(geprix) gather(geduax) gather(geprid)...
	            gather(geduad) gather(grho) gather(gsigma) tk]];
	        figure(6);
	        plot(res.itstat(:,2));
	        xlabel('Iterations');
	        ylabel('Functional value');drawnow;


	        %% Debug
	        %G = gather(PzpT(gG));
	        %figure(5);
	        %tmp = squeeze(G(:,:,1,:));
	        %imdisp(dict2img(tmp)); drawnow;

        	%% Update D partially
        	gD0(:,:,:,march) = PzpT(gG);
        	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end % End chunk
        %% Debug
        D0 = gather(gD0);
        % size(D0)
		figure(5);
	    
	    imagesc(dict2im(D0)); axis equal off; colormap gray; drawnow;

        %% Update iterations
        k = k+1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end %% End main loop

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Collect the output
    res.D = gather(gD0);  
end