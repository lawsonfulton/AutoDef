%L1 Cubature weights
configs = jsondecode(fileread('configs.json'));
%'%05.f'
input_dir = '/home/lawson/Workspace/AutoDef/models/x-final/training_data/training/';
E = readDMAT([input_dir, 'energy_', num2str(1),'.dmat']);
for ii=2:500
    E_n = readDMAT([input_dir, 'energy_', num2str(ii),'.dmat']);
    E = [E E_n];
end
random_ind = datasample([1:500],500,"Replace", false);
train_ind = random_ind(1:3*end/4);
test_ind = random_ind(3*end/4+1:end);
Etrain = E(:,train_ind);
Etest = E(:,test_ind);


%energy PCA on he training set
[U, D] = eig(Etrain*Etrain');

D = diag(D);
[D, I] = sort(D, 'descend');
U = U(:, I);

explain = D./sum(D); 
ix = find(cumsum(explain) > 0.99, 1)
U = U(:, 1:ix);

%project all energy testing data into the reduced space
alpha = U'*(Etrain);

%setup constraint matrix lazily
A = zeros(ix*size(Etrain,2), size(U,1)); 

%for each sample we have
%S'*W2*S*alpha - alph
%SijW2ikSklalphalm - alphalm
%SijSklalphalmW2ik
%SijSilalphal   W2i
%Sijvi W2is
for ii=1:size(Etrain,2)
    ii
    A((ii-1)*ix+(1:ix), :) = (diag(Etrain(:,ii))*U)';
end

options = optimoptions('quadprog', 'Display', 'none');
tol = 1e-3;
f = ones(size(Etrain,1),1);
S = [speye(size(Etrain,1)), -speye(size(Etrain,1))];
fS = [f; f];

%w2 = glinprog(f, [A; -A], [tol.*ones(size(A,1),1)+alpha(:); tol.*ones(size(A,1),1)-alpha(:)], [], [], 0.*f, f, f, options);
%w2 = glinprog(fS, [A*S; -A*S], [tol.*ones(size(A,1),1)+alpha(:); tol.*ones(size(A,1),1)-alpha(:)], [], [], 0.*fS, inf.*fS, [], options);
%w2 = glinprog(fS, [],[], A*S,alpha(:), 0.*fS, [], fS, options);

%QP
%weight = 10000000;

%just iterate until you drop the testing error below threshold. 
mass = [0, sum(f)];
while (mass(2) - mass(1)) > 2
    w2 = quadprog(A'*A, -A'*alpha(:), [], [], f', 0.5*sum(mass), 0*f,  [], [], options);
    
    w2(abs(w2) < 1e-8) = 0;
    
    error = U*((U'*diag(w2)*U)\(U'*diag(w2)*Etest) - U'*Etest);
    mag = Etest; 
    relError = sqrt((error(:)'*error(:))./(size(Etest,2)*mag(:)'*mag(:)))
    sum(abs(w2) > 0)
    if relError < tol
        %relax, you don't need so much w
        mass(2) = 0.5*sum(mass);
    else
        %panic, more w!
        mass(1) = 0.5*sum(mass);
    end
end

error = U*((U'*diag(w2)*U)\(U'*diag(w2)*Etest) - U'*Etest);
mag = Etest; 
relError = sqrt((error(:)'*error(:))./(size(Etest,2)*mag(:)'*mag(:)));
cubature_weights = sum(U,1)*((U'*diag(w2)*U)\(U'*diag(w2)));
sum(cubature_weights*Etest - sum(Etest));
writeDMAT([input_dir 'cubatureweights.dmat'], cubature_weights, true);

%COMPARE LAWSON'S An08 Results
%cubacode_results_dir = '/home/vismay/Scrapts/cubacode/models/fine_beam_final/energy_model/an08/59_samples/';
%indices = readDMAT([cubacode_results_dir, 'indices.dmat']);
%cubaweights = readDMAT([cubacode_results_dir, 'weights.dmat']);
%cw = zeros(1998,1);
%cw(indices) = cubaweights;
%sum(cw'*Etest - sum(Etest))
