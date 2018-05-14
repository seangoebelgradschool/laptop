pro gaincalc, savefiles=savefiles
;Assumes the files are local. Calculates the gain using the
; flux vs noise method. You must specify read noise.

  ;read in fits images
  ;f110 = mrdfits('H4RG_R01_M01_N110.fits', /fscale)
  f09 = mrdfits('H4RG_R01_M01_N09.fits', /fscale)
  ;f10 = mrdfits('H4RG_R01_M01_N10.fits', /fscale)
  f39 = mrdfits('H4RG_R01_M01_N39.fits', /fscale)

  ;cds = f110-f09
  ;cds = f110-f39
  cds = f39-f09

  if keyword_set(savefiles) then begin
     mwrfits, cds, 'cds110_09.fits'
     mwrfits, f10-f09, 'cds10_09.fits'
  endif

  window = cds[2048+64:2048+127 , 2048+64:2048+127] ;crop

  ;read in official mask

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

  
  ;mask = mrdfits('mask.fits', 17) ;read in the 17th stripe
  ;mask_window = mask[64:127, 2048+64:2048+127] ;crop


  ;display mask and window for visual comparison
  ;!p.multi=[0,2,1]
  ;display, window, aspect=1, max=3800, min=2800, tit="Subarray"
  ;display, mask_window, aspect=1, /noerase, tit="Hubert's Mask"
  ;!p.multi=0
;stop

  ;my_mask = where(window lt 3600 and window gt 3250); 110-09
  my_mask = where(window lt 1110 and window gt 980)
  ;my_mask = where(window lt 2470 and window gt 2290); 110-39

  ;plothist, window[where(mask_window eq 0 and window lt 5000)],$
  ;          /ylog, xtit="ADU",$
  ;          ytit="N_Occurences",  bin=50, charsize=2, charthick=2
  plothist, window[my_mask], /ylog, $
            tit="Pixels Included in Gain Calculation", xtit="ADU",$
            ytit="N_Occurences",  bin=10, charsize=2, charthick=2
  stop

  window *= (0.047+1.) / (5.*0.047 + 1.)

  mean = mean(window[my_mask])
  noise_tot = stddev(window[my_mask])
 
  shot_noise = sqrt(noise_tot^2 - 6.2^2) ;remove read noise component


  print, "Shot noise:", shot_noise, "ADU?"
  print, "Read noise?: 6.2 ADU"

  
  gain = mean/(shot_noise^2)

  print, "mean [ADU]:", mean
  ;print, "stdev:", stdev
  print, "GAIN [e-/ADU]:", gain
end
