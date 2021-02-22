/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity

Author: D. Acreman, University of Exeter

Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value          ** 
  const float xmax=30.0; // Maximum x value          *** TASK 2       
  const float ymin=0.0; // Minimum y value           *          
  const float ymax=30.0; // Maximum y value         **        
  
  /* Parameters for the Gaussian initial conditions */
  const float x0=3.0;                    // Centre(x)       ** 
  const float y0=15.0;                    // Centre(y)       *** TASK 2
  const float sigmax=1.0;               // Width(x)          *        
  const float sigmay=5.0;               // Width(y)         **        
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  const int nsteps=800; // Number of time steps   [TASK 2]

  /* Velocity */
  const float vely=0.0;     /*[Task 2]*/        // Velocity in y direction
  //const float velx=1.0;   /*[TASK 2]*/

  /**[TASK 3] In order to calculate the time step we need the maximum value of velx within the domain**/
  float velx = (fricvel/k)*log(ymax/roughlen);  // Velocity in x direction  

  /* Vertical shear*/
  const float fricvel = 0.2;  // u*: Friction velcotiy (m/s)        ** 
  const float roughlen = 1.0; // z0: Roughness length (m)            *** TASK 3
  const float k = 0.41;       // k: Von Karman’s constant           ** 


  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
  

  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy) );  
  
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  //Begin timer for gaging performance
  clock_t begin = clock();

  // Display if using OpenMP or not
  #ifdef _OPENMP
    printf("Using OpenMP.\n");

    // If using OpenMP use wall time as CPU time is inaccurate
    begin = omp_get_wtime();
  #else
    printf("Not using OpenMP.\n");
  #endif


  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
  /** [TASK 1] Can be parallelised as it has no loop carried dependencies.
  *    - Each value of i will only write to it's corresponding value in x
  *    - dx does not change so can be shared between processes
  **/
  #pragma omp parallel for default (none) shared(x, dx)
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  /** [TASK 1] Similarly to Loop 1 has no loop carried depenencies
  *    - Each value of j will only write ot it's corresponding value in y
  *    - dy does not change so can be shared between processes
  **/
  #pragma omp parallel for default (none) shared(y, dy)
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
  /** [TASK 1] Can be parallelised as not dependencies
  *   - Important x2 and y2 are privately scoped as each process has it's own values for x2 and y2
  *   - As it contains a nested loop we need to use the collapse directive with a depth of 2
  **/
  #pragma omp parallel for default(none) shared(x, y, u) private(x2, y2) collapse(2)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      x2      = (x[i]-x0) * (x[i]-x0);
      y2      = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  /** [TASK 1] This cannot be parallelised as there is an output dependency.
  *     - The file must be writtnen to in order. Otherwise gnuplot cannot process it
  *     - If using another plotting system that does not require the data to be ordered it could be possible
  **/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);

  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  /** [TASK 1] This loop cannot be parallelised as the steps must happen sequentially.
  *     - Since arrays such as u and dudt are being written to each loop  there is an output dependency 
  *     - Similarly ther is a flow dependency when writing to dudt although the value of u would have 
  *     been written to in the previous loop. Another process could write to again before it is read
  *
  **/
  for (int m=0; m<nsteps; m++){     
    
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
    /** [TASK 1] Can be parallelised as it has no dependencies
    *
    **/
    #pragma omp parallel for default(none) shared(u)
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
    /** [TASK 1] Like Loop 6 also can be parallelised due to no dependencies
    *
    **/
    #pragma omp parallel for default(none) shared(u)
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    /** [TASK 1] Can be parallelised 
    *   - Although reading from u[i][j] and u[i-1][j] since it is writing to a different array there is no Flow dependency
    *   - Because ther is a nested loop we can use the collapse directive with a depth of 2. Allowing for all the iterations to be partitioned
    *
    *   [TASK 3] Since velx is now calculated for each value of y its scope becomes private
    **/
    #pragma omp parallel for default(none) shared(dudt, u, dx, dy, y) private (velx) collapse(2)
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){

        // [TASK 3] Only calculate velx for y > 0
        if (y[j] > roughlen){
          velx = (fricvel/k) * log(y[j]/roughlen);
        } else{
          velx = 0.0;
        }

	      dudt[i][j] = -velx * (u[i][j] - u[i-1][j]) / dx
	            - vely * (u[i][j] - u[i][j-1]) / dy;
        
      }
    }
    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    /** [TASK 1] This loop can be parallelised as it has no loop dependencies
    *     - Likewise to Loop8 this loop contians a nested loop so we use the collapse directive like before
    **/
    #pragma omp parallel for default(none) shared(u, dudt, dt) collapse(2)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
	      u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  } // time loop
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  /** [TASK 1] This loop cannot be parallelised due to the same reasons as Loop 4, output dependency.
  *
  **/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  /*** [TASK 4] Write array of average values of u for each value of x ***/
  FILE *avfile;
  avfile = fopen("average.dat", "w");

  for (int i = 1; i<NX+1; i++){
    float sum = 0.0;
    for (int j = 1; j < NY+1; j++){
      sum += u[i][j];
    }
    fprintf(avfile, "%g %g\n", x[i], sum/(float)NY);
  }
  fclose(avfile);

  /** Although not specified by the assessment I added this to be able to test if the parallelisation worked**/
  // Capture time again
  clock_t end = clock();

  // Calculate time taken to execute
  double execution_time;
  #ifdef _OPENMP
    end = omp_get_wtime();
    execution_time = end - begin;
  #else
    execution_time = (double)(end - begin) / CLOCKS_PER_SEC;
  #endif

  printf("Time elapsed: %f seconds\n", execution_time);


  return 0;
}

/* End of file ******************************************************/
