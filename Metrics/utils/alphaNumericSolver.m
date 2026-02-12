function alpha = alphaNumericSolver(M,P,R,r)

% % Trapezoidal Method:
dalpha = (G*M./c^2+4*pi*G*r.^3.*P./c^4)./(r.^2-2*G*M.*r./c^2);
dalpha(1) = 0;
alphaTemp = cumtrapz(r,dalpha);
C = 1/2*log(1-2*G*M(end)./r(end)/c^2);
offset = C-alphaTemp(end);
alpha = alphaTemp+offset;


% Old manual integration
%     alpha = 1/2*log(1-2*G*M(end)./r/c^2);
% 
%     for i = flip(2:(length(r)))
%         if r(i) <= R
%             dr = -(r(i)-r(i-1));
%             dalpha = (G*M(i)/c^2+4*pi*G*r(i)^3*P(i)/c^4)/(r(i)^2*(1-2*G*M(i)/r(i)/c^2));
%             alpha(i-1) = alpha(i)+dalpha*dr;
%         end
%     end

% Simpson's rule
% alpha = 1/2*log(1-2*G*M(end)./r/c^2);
% alpha = zeros(length(r),1);
% dalpha = (G*M./c^2+4*pi*G*r.^3.*P./c^4)./(r.^2-2*G*M.*r./c^2);
% dalpha(1) = 0;
% 
% for i = flip(2:(length(r)-2))
%     a = r(i);
%     b = r(i-1);
%     [~, minIdx1] = min(abs(r-(2*a+b)/3));
%     if r(minIdx1) > (2*a+b)/3
%         minIdx1 = minIdx1 - 1;
%     end
%     minIdx1 = minIdx1 + ((2*a+b)/3-r(minIdx1))/(r(minIdx1+1)-r(minIdx1));
% 
%     [~, minIdx2] = min(abs(r-(a+2*b)/3));
%     if r(minIdx2) > (2*a+b)/3
%         minIdx2 = minIdx2 - 1;
%     end
%     minIdx2 = minIdx2 + ((2*a+b)/3-r(minIdx2))/(r(minIdx2+1)-r(minIdx2));
% 
%     k1 = dalpha(i-1);
%     k2 = legendreRadialInterp(dalpha,minIdx1);
%     k3 = legendreRadialInterp(dalpha,minIdx2);
%     k4 = dalpha(i);
%     simp = (b-a)/8*(k1+3*k2+3*k3+k4);
%     alpha(i-1) = alpha(i)+simp;
% end

end
