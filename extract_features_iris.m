 function [watermark]=extract_features_iris(water_img)
%water_img=imread('001_L_01.bmp_norm.bmp');
water_img=imresize(water_img,[10,100]);
water_img=histeq(water_img);
[template1]=encode(water_img,1,18,0.5);
%figure;imshow(water_img);title('Watermark Image');
watermark='';
for i=1:10
    for j=1:200
        template1(i,j)=num2str(template1(i,j));
        watermark=strcat(watermark,template1(i,j));
        
    end
end
end

function [template] = encode(polar_array,nscales, minWaveLength,sigmaOnf)

% convolve normalised region with Gabor filters
[E0] = gaborconvolve(polar_array, nscales, minWaveLength,sigmaOnf);

length = size(polar_array,2)*2*nscales;

template = zeros(size(polar_array,1), length);

length2 = size(polar_array,2);
h = 1:size(polar_array,1);

%create the iris template

for k=1:nscales
    
    E1 = E0{k};
    
    %Phase quantisation
    H1 = real(E1) > 0;
    H2 = imag(E1) > 0;
     
    for i=0:(length2-1)
                
        ja = double(2*nscales*(i));
        %construct the biometric template
        template(h,ja+(2*k)-1) = H1(h, i+1);
        template(h,ja+(2*k)) = H2(h,i+1);
        
    end
    
end 
end
function [EO, filtersum] = gaborconvolve(im, nscale, minWaveLength,sigmaOnf)

[rows cols] = size(im);		
EO = cell(1, nscale);          % Pre-allocate cell array

ndata = cols;
if mod(ndata,2) == 1             % If there is an odd No of data points 
    ndata = ndata-1;               % throw away the last one.
end

logGabor  = zeros(1,ndata);
result = zeros(rows,ndata);

radius =  [0:fix(ndata/2)]/fix(ndata/2)/2;  % Frequency values 0 - 0.5
radius(1) = 1;

wavelength = minWaveLength;        % Initialize filter wavelength.


for s = 1:nscale,                  % For each scale.  
    
    % Construct the filter - first calculate the radial filter component.
    fo = 1.0/wavelength;                  % Centre frequency of filter.
    rfo = fo/0.5;                         % Normalised radius from centre of frequency plane 
    % corresponding to fo.
    logGabor(1:ndata/2+1) = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
    logGabor(1) = 0;  
    
    filter = logGabor;
      
    % for each row of the input image, do the convolution, back transform
    for r = 1:rows	% For each row
        
        signal = im(r,1:ndata);
        
        
        imagefft = fft( signal );
        
        
        result(r,:) = ifft(imagefft .* filter);
        
    end
    
    % save the ouput for each scale
    EO{s} = result;
    
end                                     
end

