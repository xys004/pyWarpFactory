function [g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart] = sph2cartDiag(theta,phi,g11_sph,g22_sph)

g11_cart = g11_sph; 

E = g22_sph;

if abs(phi) == pi/2
    cosPhi = 0;
else
    cosPhi = cos(phi);
end

if abs(theta) == pi/2
    cosTheta = 0;
else
    cosTheta = cos(theta);
end


g22_cart = (E*cosPhi^2*sin(theta)^2 + (cosPhi^2*cosTheta^2)) + sin(phi)^2;
g33_cart = (E*sin(phi)^2*sin(theta)^2 + (cosTheta^2*sin(phi)^2)) + cosPhi^2;
g44_cart = (E*cosTheta^2 + sin(theta)^2);

g23_cart = (E*cosPhi*sin(phi)*sin(theta)^2 + (cosPhi*cosTheta^2*sin(phi)) - cosPhi*sin(phi));
g24_cart = (E*cosPhi*cosTheta*sin(theta) - (cosPhi*cosTheta*sin(theta)));
g34_cart = (E*cosTheta*sin(phi)*sin(theta) - (cosTheta*sin(phi)*sin(theta)));

% g22_cart = (E*cos(phi)^2*sin(theta)^2 + (cos(phi)^2*cos(theta)^2)) + sin(phi)^2;
% g33_cart = (E*sin(phi)^2*sin(theta)^2 + (cos(theta)^2*sin(phi)^2)) + cos(phi)^2;
% g44_cart = (E*cos(theta)^2 + sin(theta)^2);
% 
% g23_cart = (E*cos(phi)*sin(phi)*sin(theta)^2 + (cos(phi)*cos(theta)^2*sin(phi)) - cos(phi)*sin(phi));
% g24_cart = (E*cos(phi)*cos(theta)*sin(theta) - (cos(phi)*cos(theta)*sin(theta)));
% g34_cart = (E*cos(theta)*sin(phi)*sin(theta) - (cos(theta)*sin(phi)*sin(theta)));

end
