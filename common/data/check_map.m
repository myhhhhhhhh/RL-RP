clear;

load map_matrix.mat
load dis_matrix.mat

[rs, cs] = size(map_matrix);
for x=1:1:rs
    y = find(map_matrix(x, :)==1)-1;
    [x-1, y]
end

for xx=1:1:rs
    y = find(dis_matrix(xx, :)~=0);
    [xx-1, dis_matrix(xx, y)]
end

