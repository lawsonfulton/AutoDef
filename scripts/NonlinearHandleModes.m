%% Nonlinear Handle Mode Sampling %%

% TODO (for starters, not in order of importance)
% 1) Fix computing of alpha so scaling the mesh is unnecessary (related to
% item 6)
%
% 2) Fix getStrainEnergyPerElement so that energy vectors are saved as well
%
% 3) Support different material properties that get specified in a config file
% ie. modify configs/x-config.json so that it has a YM and poisson value
%
% 4) Don't oversample the rest pose (only final pose and introduce random
% termination)? This might not work because we may need intermediary
% exampls, also wastes a lot of compute, so it should be turned on/off with
% flag
%
% 5) Make it a lot more modular, so it takes as input the json config file
% and outputs all of the training data and a parameters.json that saves the
% material config
%
% 6) expose all algorithm parameters (and give them more understandable names)
% and determine which ones depend on the input, such as alpha.


function NonlinearHandleModes(yield)

num_samples = 100;
yield = 1.6e4;

out_dir = './test_run3/';
if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end

mesh_path = '../meshes/Fine_beam.mesh';

scale = 0.1; 
[V, T, F] = readMESH(mesh_path);
V = V * scale;


%boundary conditions
%Choose Handle Points
H = (abs(V(:,1) - max(V(:,1))) < 0.01);
handle_indices = find(H);
nH = sum(H);

%re arrange vertices
%V = [V(H,:); V(~H,:)];
[V,~,~,IM] = faces_first(V,[],handle_indices);
F = IM(F);
T = IM(T);

%build Mesh
% DT = delaunayTriangulation(V(:,1), V(:,2), V(:,3));
% T = DT.ConnectivityList; 

% F = boundary_faces(T);
 
%Boundary Conditions
leftV = (abs(V(:,1) - min(V(:,1))) < 0.01);
P = fixedBC(leftV);

%Fixed boundary constraint matrix
J = kron(diag(leftV), eye(3));
J(sum(abs(J),2) == 0,:) = [];
b = zeros(size(J,1),1);

Evec = [];
tot_its = 0;
while tot_its < num_samples
    
    %Build Stiffness Matrix
    fem = WorldFEM('neohookean_linear_tetrahedra', V, T);

    %% Solve for Modes %%
    rng('shuffle')
    h = zeros(3*nH,1); %handle displacements
    u = zeros(numel(V),1);
    alpha = 0.1%0.5 %0.1 %step length % set based on size of bounding box?
    step_size = 0.05 % 0.1 %0.05
    numIter = randi([1 30],1,2);

    Rh = eye(3,3);
    th = zeros(3,1);


    K = P*stiffness(fem)*P';
    %form handle tangent stiffness matrix via condensation
    %Condensation
    Khh = K(1:3*nH, 1:3*nH);
    Khu = K(1:3*nH, (3*nH+1):end);
    Kuu = K((3*nH+1):end,(3*nH+1):end);
    Kstar = full(Khh - (Khu*(Kuu\Khu')));
    [MODE, VAL] = eig(-Kstar); 

    %mode wrangling
    [~, I] = sort(abs(diag(real(VAL))), 'ascend');
    val = real(diag(VAL));
    val = val(I);

    fracture = false;


    for ii=1:numIter(1)


        %grab handle position from modes and use as constraint
        if(or(rand > 0.5, ii == 1))
            modeSelect = int32(rand()*5) + 1;%rigid of the handle only 
        else
            weights = 1./abs(val);
            weights = weights./sum(weights);
            modeSelect = randsample(1:6, 1, true, weights(1:6));
        end

        s = sign(rand - 0.5);

        for jj= 1:numIter(2)

            disp([num2str(alpha), ' ' num2str(ii),':',num2str(numIter(1)),' ',num2str(jj),':',num2str(numIter(2))]);
            MODE = MODE(:, I);

            %don't go backwards
            if(jj > 1) 
                s  = s*sign(prevMode'*MODE(:,modeSelect));
            end
            oldH = V(1:nH,:) + reshape(h, 3, nH)';

            nSteps = ceil(alpha/step_size);
            h0 = h;
            for kk = linspace(0, alpha, nSteps)
                h = h0 + s*MODE(:,modeSelect)*kk;
                newH = V(1:nH,:) + reshape(h, 3, nH)';
                [R,t] = fit_rigid(newH, oldH);

                %replace newH with rigidly transformed oldH
                Rh = R'*Rh;
                th = R'*th + t';
                h = reshape(Rh*V(1:nH,:)' + th - V(1:nH,:)', 3*nH,1);

                %number of steps 
                u = minEnergyConfig(fem,u,[J; speye(3*nH, numel(V))], [b; h]);
                oldH = V(1:nH,:) + reshape(h, 3, nH)';
            end

            prevMode = MODE(:, modeSelect);
            setQ(fem,u);
            K = P*stiffness(fem)*P';

            %form handle tangent stiffness matrix via condensation
            %Condensation
            Khh = K(1:3*nH, 1:3*nH);
            Khu = K(1:3*nH, (3*nH+1):end);
            Kuu = K((3*nH+1):end,(3*nH+1):end);
            Kstar = full(Khh - (Khu*(Kuu\Khu')));
            [MODE, VAL] = eig(-Kstar); 

            %mode wrangling
            [~, I] = sort(diag(real(VAL)), 'ascend');

            val = real(diag(VAL));
            val = val(I);

            u1 = reshape(u, 3, size(V,1))';
            trisurf(F, V(:,1) + u1(:,1),V(:,2) + u1(:,2),V(:,3) + u1(:,3));
            drawnow
            axis equal
            axis vis3d
            img = getframe(gcf);
            %imwrite(img.cdata, ['test', num2str(num),'_',num2str(ii),'_',num2str(jj),'.png']);

            imwrite(img.cdata, [out_dir, 'displacements_image_', num2str(tot_its,'%05.f'),'.png']);
            writeDMAT([out_dir, 'displacements_', num2str(tot_its,'%05.f'),'.dmat'], u1, false);
            tot_its = tot_its + 1;
            %save out per tet potential energies as well. 
            %Evec = [Evec strainEnergyPerElement(fem)];
            writeDMAT([out_dir, 'energy_', num2str(tot_its, '%05.f'),'.dmat'], Evec, false);

            S = stress(fem,u);
            disp(max( sum(S.*S, 2)))

            if(max(sum(S.*S, 2) > yield) == 1)
                fracture = true;
            end

            if(fracture) 
                disp(['FRACTURE'])
                break;
            end
        end

        if(fracture)
            disp(['FRACTURE']);
            break;
        end
    end

    %draw the result
    %u = reshape(u, 3, size(V,1))';
    %tetramesh(T, V + u)
    clear fem;
end

%writeDMAT('energyVecs.dmat', Evec);

%% non-linear constraint on magnitude 
function [C,Ceq, DC, DCeq] = constraints(x) 
    C = [];
    Ceq = x'*x-25; 
    
    if nargout > 2
        DC = [];
        DCeq = 2*x;
    end
end

function h = hessinterior(x,lambda, fem)
    
    setQ(fem, x);
    h = -stiffness(fem);
end

%energy and gradient
function [c, g] = energy(x, fem) 
    c = strainEnergyFromQ(fem,x);
    
    if nargout > 1 % gradient required
        g = internalForce(fem);
        g = -g;
    end
end

%% Static Solve Given a BC %%
function u = minEnergyConfig(fem, u, J,b)  

    %use incremental loading
    options = optimoptions('fmincon','Display','none', 'MaxFunctionEvaluations', 10000000000, 'MaxIterations', 10000, 'Algorithm', 'interior-point', 'GradObj', 'On', 'Hessian', 'user-supplied', 'HessianFcn',@(x,lambda) hessinterior(x,lambda,fem), 'DerivativeCheck','off');
    u = fmincon(@(x) energy(x,fem), u, [],[], J,b, [], [], [], options);
end

function P = fixedBC(bc)
    n = 1:numel(bc);
    P = sparse(kron(sparse(n,n, abs(1-bc)), sparse(eye(3,3))));
    P(sum(P,2) == 0, :) = [];
end

end

