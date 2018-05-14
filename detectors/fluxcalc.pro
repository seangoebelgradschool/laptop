pro fluxcalc, checkhist=checkhist, flux=flux, dark=dark, useref=useref
;Calculates the variance of different sizes of CDS deltas
; and computes the gain.
;If given appropriate keyword, gives flux as function of
; CDS delta.

  if keyword_set(dark) then begin
     print, "Working on dark frames."
     dir = 'dark_4/'
     files = (indgen(52)+1)*10; if running on everything
     ;files = indgen(30)+10
     f9 = mrdfits(dir+'H4RG_R05_M01_N09.fits', /fscale, /silent)
  endif else begin
     print, "Working on illuminated frames."
     dir = 'illum_4/'
     files = indgen(30)+10
     ;files = [indgen(30)+10 , indgen(20)*5+40] ; if running on everything
     f9 = mrdfits(dir+'H4RG_R01_M01_N09.fits', /fscale, /silent)
  endelse

  
  if keyword_set(flux) then begin
     print, "Calculating flux."
     flux=fltarr(n_elements(files))
  endif else begin
     print, "Calculating noise."
     noise = fltarr(n_elements(files))
  endelse

  for i = 0, n_elements(files)-1 do begin
     print, "Reading frame"+strcompress(files[i])
     if keyword_set(dark) then begin
        frame = mrdfits(dir+'H4RG_R05_M01_N'+strcompress(files[i], /remove)+ $
                        '.fits', /fscale, /silent)
     endif else begin
        frame = mrdfits(dir+'H4RG_R01_M01_N'+strcompress(files[i], /remove)+ $
                        '.fits', /fscale, /silent)
     endelse

     ;take a chunk
     crop = (frame-f9)[2048+64:2048+127 , *];2048+64:2048+127]
     ;crop = (frame-f9)[2048:2048+127 , 2048:2048+127]

     if keyword_set(useref) then begin
        for j = 0, (size(crop))[1]-1 do begin
           ;subtract off the reference pixels
           crop[j,*] -= mean((frame-f9)[2048+64+j , [0,1,2,3,4092,4093,4094,4095]])
        endfor
     endif

     if keyword_set(flux) then begin
        ;if keyword_set(dark) then 
        flux[i] = median(crop)

        ;if files[i] mod 30 eq 0 then begin
           ;lowend  = (crop[sort(crop)] )[n_elements(crop)*0.001]
           highend = (crop[sort(crop)] )[n_elements(crop)*0.995]

           if i ne 0 then plothist, crop, /ylog, bin=10, yrange=[0.1, 10.^3.], $
                     xrange=[min(crop), highend]
           vline, flux[i]
           ;stop
        ;endif

     endif else begin
        ;values are ideal for H4RG illuminated data
        ;lowend  = (crop[sort(crop)] )[n_elements(crop)*0.04]
        ;highend = (crop[sort(crop)] )[n_elements(crop)*0.995]

        ;These values are better for dark data
        lowend  = (crop[sort(crop)] )[n_elements(crop)*0.01]
        highend = (crop[sort(crop)] )[n_elements(crop)*0.97]

        mymask = where(crop gt lowend and crop lt highend)

        if keyword_set(checkhist) then begin
           !p.multi=[0,1,2]
           if files[i] mod 30 eq 0 then begin
              plothist, crop, bin=3, /ylog
              plothist, crop[mymask], bin=3, /ylog
              ;stop
           endif
           !p.multi=0
        endif
        
        noise[i] = stddev(crop[mymask])^2
        
     endelse
  endfor

  !x.style=1
  !y.style=1

  if keyword_set(flux) then begin
     plot, files-9, flux, tit="Flux vs. CDS Frame Delta", $
           xtit="Number of Frames Difference in CDS Frame", $
           ytit="Flux [Median ADU per pixel]", charsize=2, $
           xrange=[0, max(files)-9+1], $
           yrange=[min(flux)-0.05*abs(min(flux)), max(flux)*1.05], psym=2

     result = poly_fit(files-9, flux, 1)
     oplot, files-9, result[0] + result[1] * (files-9.); + $
            ;result[2] * (files-9.)^2
     legend, linestyle=[0], ['y='+ $
                             ;strmid(strcompress(result[2],/remove),0,6) + $
                             ;textoidl('x^2') + '+' + $
                             strmid(strcompress(result[1],/remove),0,4) + $
                             'x+' + $
                             strmid(strcompress(result[0],/remove),0,4)], $
             charsize=2, /top, /left

     print, "The dispersion of the data about the best fit line is:", $
            stddev(flux - (result[0] + result[1] * (files-9.)))

  endif else begin
     plot, files-9, noise, tit="Pixel Variance vs. CDS Frame Delta",$
           xtit="Number of Frames Difference in CDS Frame", $
           ytit="Variance [ADU^2]", charsize=2, $
           xrange=[0, max(files)-9+1], yrange=[0, max(noise)*1.05], psym=2
     
     result = poly_fit(files-9,noise, 2)
     oplot, files-9, result[0] + result[1] * (files-9.) +$
            result[2] * (files-9.)^2 ;+$
            ;result[3] * (files-9.)^3
     legend, linestyle=[0], ['y='+$
                             ;strmid(strcompress(result[3],/remove),0,7) + $
                             ;textoidl('x^3') + '+' + $
                             strmid(strcompress(result[2],/remove),0,6) + $
                             textoidl('x^2') + '+' + $
                             strmid(strcompress(result[1],/remove),0,4) + $
                             'x+' + $
                             strmid(strcompress(result[0],/remove),0,4)], $
             charsize=2, /top, /left
     
     print, "Y-intercept (read noise:):", result[0]
  endelse


 stop
end
