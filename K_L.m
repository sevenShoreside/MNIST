function [X_new U] = K_L(X,num)
N=size(X);
C=cov(X');
[eigenvector,eigenvalue]=eig(C);
U = eigenvector(:,N(1)-num+1:N(1));
X_new = U' * X;
end


