function vStates = rpi_apply_rescaling( cStates, cRescaling )

lD=diag(cRescaling);

vStates = cStates*lD;

return;