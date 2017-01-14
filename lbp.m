function [ lbp_frame ] = lbp( frame )
%lbp Local Binary Patterns

    [r, c] = size(frame);
    lbp_frame = zeros(r-2,c-2);
    for i = 2:r-1
        for j = 2:c-1
            lbp_val = 0;
            bit = 1;
            for m = -1:1
                for n = -1:1
                    if not(m==0 && n==0)
                        if frame(i,j) < frame(i+n,j+m)
                            lbp_val = bitset(lbp_val,bit);
                        end;
                        bit = bit+1;
                    end;
                end;
            end;
            lbp_frame(i-1,j-1) = lbp_val;
        end;
    end;
end
