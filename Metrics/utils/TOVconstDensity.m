function P = TOVconstDensity(R,M,rho,r)
    P = c^2*rho.*((R*sqrt(R-2*G*M(end)/c^2)-sqrt(R^3-2*G*M(end).*r.^2/c^2))./(sqrt(R^3-2*G*M(end).*r.^2/c^2)-3*R*sqrt(R-2*G*M(end)/c^2))).*(r<R);
end