function [ y_out, y_outv, w_out, logtrain, logvalid, log_max, index] = nn_train_prob_es( x, y, w, H, M, alpha, lr, its, xv, yv)
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

logtrain = zeros(its, 1);
logvalid = zeros(its,1);

log_max = -10000;

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
        g(:,j) = 1./(1+exp(-xin(:,j)));
        dEdx(:,j) = (g(:,j)-y);
        
        xinv(:,j) = g2*wlm(:,j);
        g2(:,j) = 1./(1+exp(-xinv(:,j)));
    end
    
    for j = H+I:-1:I+1
        dEdx(:, j) = dEdx(:, j+1:I+H+M) * wlm(j, j+1:I+H+M)' .* g(:, j) .* (1 - g(:,j));
    end
    
    dwlm = g' * dEdx;
    w_out = wlm;
    wlm = wlm.*alpha - lr * dwlm .* (wlm ~= 0); 
    y_out = g(:, I+H+1:I+H+M);
    %error(i) = sum(abs(y-round(y_out)))/t;
    
    y_outv = g2(:, I+H+1:I+H+M);
    %errorv(i) = sum(abs(yv-round(y_outv)))/t2;

    logtrain(i) = y'*log(y_out) + (1 - y)'*log(1-y_out);
    logtrain(i) = logtrain(i)/t;

    logvalid(i) = yv'*log(y_outv) + (1 - yv)'*log(1-y_outv);
    logvalid(i) = logvalid(i)/t2;
    
    if i >2000 && log_max > logvalid(i)
        w_out = w_prev;
        break;
    else
        w_prev = w_out;
        log_max = logvalid(i);
        index = i;
    end
    
end

end

