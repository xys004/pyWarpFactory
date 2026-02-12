function f = compactSigmoid(r,R1,R2,sigma,Rbuff)

f = abs(1./(exp(((R2-R1-2*Rbuff)*(sigma+2))/2*(1./(r-R2+Rbuff)+1./(r-R1-Rbuff)))+1).*(r>R1+Rbuff).*(r<R2-Rbuff)+(r>=R2-Rbuff)-1);
if any(isinf(f)) || any(~isreal(f))
    error('compact sigmoid returns non-numeric values!')
end

end