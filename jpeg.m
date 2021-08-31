best_diff = 0;
temp = f;
best_k = 1;
best_m = 1;
comp_k = 1;
comp_m = 1;
comp_comp = 0;
f_jpeg_comp = f;
for k = 1:8
    for m = 1:8
        ft = temp-128;
        a = C*ft*C';
        l=zeros(8,8);
        b=l;
        for i = 1:8
            for j = 1:8
                l(i,j) = floor(a(i,j)/q(i,j) + 1/2);
                b(i,j) = l(i,j)*q(i,j);
            end
        end
        i_comp = 8;
        j_comp = 8;
        comp=0;
        step = 1;
        while 1
            if b(i_comp,j_comp)==0
                comp = comp + 1;
            else
                break;
            end
            if i_comp == 1 && j_comp == 2
                break;
            end
            if (i_comp == 8 && step == 1) || (i_comp == 1 && step == -1)
                j_comp = j_comp - 1;
                step = -step;
            elseif (j_comp == 8 && step == -1) || (j_comp == 1 && step == 1)
                i_comp = i_comp - 1;
                step = -step;
            else
                i_comp = i_comp + step;
                j_comp = j_comp - step;
            end
        end
        ft_jpeg = C'*b*C;
        f_jpeg = floor(ft_jpeg + 128.5);
        for i = 1:8
            for j = 1:8
                if f_jpeg(i,j)>255
                    f_jpeg(i,j) = 255;
                elseif f_jpeg(i,j)<0
                    f_jpeg(i,j) = 0;
                end
            end
        end
        f_jpeg = circshift(f_jpeg, [8-k+1 8-m+1]);
        diff = norm(f_jpeg - f, 'fro');
        if (k==1 && m==1) || diff<best_diff
            best_diff = diff;
            best_comp = comp;
            best_k = k;
            best_m = m;
            f_jpeg_best = f_jpeg;
            if k==1 && m==1
                prev_diff = diff;
                f_jpeg_prev = f_jpeg;
                prev_comp = comp;
            end
        end
        if comp > comp_comp
            comp_comp = comp;
            comp_diff = diff;
            comp_k = k;
            comp_m = m;
            f_jpeg_comp = f_jpeg;
        end 
        temp = circshift(temp, [0 1]);
    end
    temp = circshift(temp, [1 0]);
end

for i = 1:3
    subplot(2,3,i);
    imshow(f,[]);
end
subplot(2,3,2);
title('original');
subplot(2,3,4);
imshow(f_jpeg_prev, []);
title(["prev diff = " + prev_diff, "prev comp = " + prev_comp]);
subplot(2,3,5);
imshow(f_jpeg_best, []);
title(["best diff = " + best_diff, "best comp = " + best_comp]);
subplot(2,3,6);
imshow(f_jpeg_comp, []);
title(["comp diff = " + comp_diff, "comp comp = " + comp_comp]);




