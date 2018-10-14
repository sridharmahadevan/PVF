/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2000-2002 
%
% Michail G. Lagoudakis (mgl@cs.duke.edu)
% Ronald Parr (parr@cs.duke.edu)
%
% Department of Computer Science
% Box 90129
% Duke University
% Durham, NC 27708
% 
%
% xdot = pendulum(t, x) 
%
% C implementation of the equation of the pendulum
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/

#include "stdafx.h"


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <limits.h>
#include <signal.h>
/*#include <sys/times.h>
#include <sys/time.h> */
#include <errno.h>

#define CCm 2.0 		// Mass of the pendulum
#define CCM 8.0 		// Mass of the cart
#define CCl 0.5 	        // Length of the pendulum
#define CCg 9.8 		// Gravity constant
#define CCa ( 1.0 / ( CCm + CCM ) ) 


void penddot(double *xdot, double t, double *x)
{
  double u;
  double cx;
 
  // Nonlinear model 
  
  u = x[2]; 
  cx = cos(x[0]);

  xdot[0] = x[1]; 
  xdot[1] = ( CCg * sin( x[0] ) - 
	      CCa * CCm * CCl * (x[1])*(x[1]) * sin(2.0*x[0]) / 2.0 -
 	      CCa * cos(x[0]) * u ) / 
    ( 4.0 / 3.0 * CCl - CCa * CCm * CCl * cx * cx ); 
  xdot[2] = 0;

  return;
  
}





#include "mex.h"

/* Input Arguments */

#define	T_IN	prhs[0]
#define	X_IN	prhs[1]


/* Output Arguments */

#define	XD_OUT	plhs[0]


void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray *prhs[] )
     
{ 

  double *xdot;
  double *x;
  double *t;
 
  /* Check for proper number of arguments. */
  if(nrhs!=2) {
    mexErrMsgTxt("Two inputs required.");
  } else if(nlhs>1) {
    mexErrMsgTxt("Too many output arguments");
  }
  
  /* Create a matrix for the return argument */ 
  XD_OUT = mxCreateDoubleMatrix(3, 1, mxREAL); 
  
  /* Assign pointers to the various parameters */ 
  xdot  = mxGetPr(XD_OUT);
  
  t = mxGetPr(T_IN); 
  x = mxGetPr(X_IN);
  
  /* Do the actual computations in a subroutine */
  penddot(xdot, *t, x);

  return;
    
}


