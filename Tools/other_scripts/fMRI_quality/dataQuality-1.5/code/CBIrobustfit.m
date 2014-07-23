function [b,w] = CBIrobustfit(X,y)
%
% [b,w] = CBIrobustfit(X,y);
%
% This function performs a robust fit to the linear system  y = X*b.
%   It is equivalent to Matlab's robustfit function
%   (default options, with no constant term).
%
% Written by Pablo Velasco
%
% PJV: v.1.1: Jan 2014

% To Do:
%   - make the tolerance and max_iterations optional inputs
%   - add a debug flag input to control printing

% PJV: v.1.1: Jan 2014
%   Uses    temp2\temp3  , rather than    inv(temp2)*temp3   .
%   Initializes "myErr" before using it inside the loop.
% PJV: v.1.0: Nov 2006


myTolerance = 10^-4;  % Good enough for us
max_iter = 50;      % Maximum number of iterations allowed

% Initialize variables:
w = ones(size(y));         % weights
b = zeros(size(X,2),1);    % estimator
myErr = zeros(size(y));    % error
for iter = 1:max_iter
   W = diag(w);
%%% PJV: This code is highly inefficient when compiled!
%     b = inv(X'*W*X)*X'*W*y;
%%% PJV/
   temp1 = X'*W;
   temp2 = temp1*X;
   temp3 = temp1*y;
   b = temp2\temp3;
   e = y - X*b;
%     H = X*inv(X'*W*X)*X'*W;
%     e = (e\sqrt(1-H))';

    myErr(iter) = e'*W*e;

    sigma = median(abs(e))/0.6745;
    k = 4.685*sigma;
    w = zeros(size(e));
    inside = find(abs(e)<=k);
    w(inside) = (1-(e(inside)/k).^2).^2;
%    V = inv(X'*W*X)*sigma^2;
%    myErr(iter) = norm(V);

%%% PJV: Printing, for debugging:
% %     fprintf('Iter = %d, norm(e) = %f x 10^-3, myErr = %f x 10^-6, b(1) = %f,\t b(2) = %f\n',iter,norm(e)*1000,myErr(iter)*1000000,b(1),b(2));
%     if length(b)==1
% %      fprintf('Iter = %d, norm(e) = %f x 10^-3, myErr = %f x 10^-6, Frobenius-norm = %f x 10^-6, b = %f\n',iter,norm(e)*1000,myErr(iter)*1000000,norm(V,'fro')*1000000,b);
%       fprintf('Iter = %d, myErr = %f x 10^-6, b = %f\n',iter,myErr(iter)*1000000,b);
%     elseif length(b)==2
% %          fprintf('Iter = %d, norm(e) = %f x 10^-3, myErr = %f x 10^-6, Frobenius-norm = %f x 10^-6, b(1) = %f,\t b(2) = %f\n',iter,norm(e)*1000,myErr(iter)*1000000,norm(V,'fro')*1000000,b(1),b(2));
%       fprintf('Iter = %d, myErr = %f x 10^-6, b(1) = %f,\t b(2) = %f\n',iter,myErr(iter)*1000000,b(1),b(2));
%     end
%%% PJV/

    % Check for convergence:
    if (iter>2) && ...
            (abs(myErr(iter-1) - myErr(iter))<myTolerance*myErr(iter))
        break
    end

end

%%% PJV: for debugging purposes:
% if iter == max_iter
%   fprintf('I reached max number of iterations.  %f parameters\n', length(b))
% %   figure; plot(log10([abs(myErr(2:end)-myErr(1:end-1))./myErr(1:end-1); myTolerance*ones(1,length(myErr)-1)])')
% %   figure; plot(X(:,1), [y X*b])
% end
% 
% figure; subplot(1,2,1); plotyy(X(:,1), [y X*b],X(:,1),diag(W));
% subplot(1,2,2); plotyy(X(:,1), e, X(:,1), w);
%%% PJV/ 
