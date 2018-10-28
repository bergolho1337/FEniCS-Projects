#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

static constexpr double D0 = 0.05;
static constexpr double TAU = 1.0;
static constexpr double w0 = 0.05;
static constexpr double w1 = 0.0;

struct point
{
    double x, y, z;
};

struct element
{
    int ori, dest;
};

double f_bar ()
{
    return 6.0*(w0 - w1);
}

double v ()
{
    return sqrt(2.0*D0)*f_bar() / TAU;
}

double f (const double x, const double t)
{
    return (1.0 - tanh( (x - v() * t)/(2.0 * sqrt(2.0*D0)) )) / 2.0;
}

void write_solution_to_vtk (const double *u,\
                            const int xelem, const int telem,\
                            const double dx, const double dt,\
                            const double xmin, const double tmin)
{
    // Create points
    struct point *points = new struct point[xelem];
    for (int i = 0; i < xelem; i++)
    {
        points[i].x = xmin + i*dx;
        points[i].y = 0.0;
        points[i].z = 0.0;
    }
    // Create elements
    struct element *elements = new struct element[xelem-1];
    for (int i = 0; i < xelem-1; i++)
    {
        elements[i].ori = i;
        elements[i].dest = i+1; 
    }

    // For each timestep write a VTK file
    for (int k = 0; k < telem; k++)
    {
        FILE *file;
        char filename[50];
        
        // Write the transmembrane potential
        sprintf(filename,"vtk/sol%d.vtk",k);
        file = fopen(filename,"w+");
        fprintf(file,"# vtk DataFile Version 3.0\n");
        fprintf(file,"Monodomain MVF\n");
        fprintf(file,"ASCII\n");
        fprintf(file,"DATASET POLYDATA\n");
        fprintf(file,"POINTS %d float\n",xelem);
        for (int i = 0; i < xelem; i++)
        {
            fprintf(file,"%g %g %g\n",points[i].x,points[i].y,points[i].z);
        }
        fprintf(file,"LINES %d %d\n",xelem-1,(xelem-1)*3);
        for (int i = 0; i < xelem-1; i++)
            fprintf(file,"2 %d %d\n",elements[i].ori,elements[i].dest);

        fprintf(file,"POINT_DATA %d\n",xelem);
        fprintf(file,"SCALARS vm float 1\n");
        fprintf(file,"LOOKUP_TABLE default\n");
        for (int i = 0; i < xelem; i++)
            fprintf(file,"%g\n",u[k*xelem+i]);
        fclose(file);
    }
    
    delete [] points;
    delete [] elements;
}

int main ()
{
    const double xmin = 0.0;
    const double xmax = 4.0;
    const double tmin = 0.0;
    const double tmax = 100.0;

    const int xelem = 50;
    const int telem = 50;

    double dx = (xmax - xmin) / xelem;
    double dt = (tmax - tmin) / telem;
    double *u = new double[xelem*telem];

    for (int i = 0; i < telem; i++)
    {
        double t = tmin + i*dt;
        for (int j = 0; j < xelem; j++)
        {
            double x = xmin + j*dx;
            u[i*xelem+j] = f(x,t);
        }
    }

    write_solution_to_vtk(u,xelem,telem,dx,dt,xmin,tmin);

    delete [] u;

    return 0;
}