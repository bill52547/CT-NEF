function [ img ] = CTbackprojection( proj, param )
%CTBACKPROJECTION Summary of this function goes here
%   Detailed explanation goes here

img = zeros(param.nx, param.ny, param.nz, 'single');
h = waitbar(0, 'backprojecting');
for i = 1:param.nProj  
    i / param.nProj
%     disp(['backprojecting view = ' num2str(i)])
    waitbar(i / param.nProj, h, num2str(i / param.nProj));
    img = img + backprojection(proj(:,:,i),param,i);
end
delete(h);

end

