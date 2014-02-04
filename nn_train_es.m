function [ y_out, y_outv, w_out, error, errorv, errormin, index] = nn_train_es( x, y, w, H, M, lr, its, xv, yv)
% Neural Net code based on example code shown in class

t = size(x,1);
I = size(x,2) + 1;
xin = zeros(t, I+H+M);
xin(:,1) = 1;
xin(:,2:I) = x;

g = zeros(t, I+H+M);
g(:,1) = 1;
g(:,2:I) = x;

t2 = size(xv,1);
g2 = zeros(t2, I+H+M);
g2(:,1) = 1;
g2(:,2:I) = xv;

xinv = zeros(t, I+H+M);
xinv(:,1) = 1;
xinv(:,2:I) = xv;

error = zeros(its, 1);
errorv = zeros(its, 1);
wlm = w;

dEdx = zeros(t, I+H+M);
w_prev = w;
errormin = 100;
index = 0;
%loop for number of iterations
for i = 1:its
    
    for j =I+1:I+H
        xin(:,j) = g*wlm(:,j);        
        g(:,j) = 1./(1+exp(-xin(:,j)));
        
        g2(:,j) = 1./(1+exp(-xinv(:,j)));
        xinv(:,j) = g2*wlm(:,j);
    end
    
    for j = I+H+1:I+H+M
        xin(:,j) = g*wlm(:,j);
        g(:,j) = xin(:,j);
        dEdx(:,j) = 2*(g(:,j)-y);
        
        xinv(:,j) = g2*wlm(:,j);
        g2(:,j) = xinv(:,j);
    end
    
    for j = H+I:-1:I+1
        dEdx(:, j) = dEdx(:, j+1:I+H+M) * wlm(j, j+1:I+H+M)' .* g(:, j) .* (1 - g(:,j));
    end
    
    dwlm = g' * dEdx;
    w_out = wlm;
    wlm = wlm - lr * dwlm .* (wlm ~= 0); 
    y_out = g(:, I+H+1:I+H+M);
    error(i) = sum((y-y_out).^2)/t;
    
    y_outv = g2(:, I+H+1:I+H+M);
    errorv(i) = sum((yv-y_outv).^2)/t2;
    
    if i >10000 && errormin < errorv(i)
        w_out = w_prev;
        break;
    else
        w_prev = w_out;
        errormin = errorv(i);
        index = i;
    end
    
    
end

end