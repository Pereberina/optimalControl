#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#define EPSILON 0.000000005
//#define STOPPING_RULE(prev, next) (fabs(prev - next) <= EPSILON)
#define EPSILON 0.01 // accuracy
#define MIN -14.0/3
#define STOPPING_RULE(next) (fabs(next - MIN) <= EPSILON)

#define L 10.0 // Lipschitz constant
#define T 4.0 // time limit
#define SIZE ((int)(T/pow(EPSILON, 3.0/2))) // size of step + 1
#define TAU (T/(SIZE - 1)) // step in time
#define F0(t, x, u) (u*u + x) // integrand
#define F(t, x, u) (u)
#define GRADIENT_FI(x) 0 // gradient of a terminal part
#define X0 0 // initial condition
#define CONST 0.5 // y0

// H is Pontryagin's function
#define H_x(t, x, u, psi) (1)
#define H_u(t, x, u, psi) (2*u + psi)


// set time values
void set_t(double *t) {
    int k;

    for (k = 0; k < SIZE; k++) {
        t[k + 1] = t[k] + TAU;
    }
}

// q should be in U
// adjust values otherwise
void accept_value(double *q) {
    int k;

    // U = [-1,1]
    for (k = 0; k < SIZE; k++) {
        if (q[k] < -1) {
            q[k] = -1;
            continue;
        }
        if (q[k] > 1) {
            q[k] = 1;
        }
    }
}


int gradient_J(double *u, double *x, double *t, double *gradient) {
    int k;
    double *psi;

    if ((psi = (double *)malloc(SIZE*sizeof(double))) == NULL) {
        printf("Not enough memory\n");
        return -1;
    }

    // lattice
    x[0] = X0;
    for (k = 0; k < SIZE - 1; k++) {
        x[k + 1] = x[k] + TAU*F(t[k], x[k], u[k]);
    }

    psi[SIZE - 1] = GRADIENT_FI(x[SIZE - 1]);
    for (k = SIZE - 2; k >= 0; k--) {
        psi[k] = psi[k + 1] + TAU*H_x(t[k + 1], x[k + 1], u[k + 1], psi[k + 1]);
    }

    for (k = 0; k < SIZE; k++) {
        gradient[k] = H_u(t[k], x[k], u[k], psi[k]);
    }

    free(psi);

    return 0;
}


// calculate integral via rectangle method
double J(double *u, double *x, double *t) {
    double S;
    int k;

    S = 0.0;

    for (k = 0; k < SIZE; k++) {
        S += TAU*F0(t[k], x[k], u[k]);
    }

    return S;
}

int main(void) {
    double A[2];
    double alpha[2];
    double *u[2];
    double *y[2];
    double *q[2];
    double *x;
    double *gradient;
    double *t;
    int prev;
    int next;
    int k;
    int iterations;
    double J_prev, J_next;
    struct timespec start, end;


    if ((t = (double *)malloc(SIZE*sizeof(double))) == NULL) {
        printf("Not enough memory\n");
        return -1;
    }

    if ((x = (double *)malloc(SIZE*sizeof(double))) == NULL) {
        printf("Not enough memory\n");
        return -1;
    }

    if ((gradient = (double *)malloc(SIZE*sizeof(double))) == NULL) {
        printf("Not enough memory\n");
        return -1;
    }
    for (k = 0; k < 2; k++) {
        if ((u[k] = (double *)malloc(SIZE*sizeof(double))) == NULL) {
            printf("Not enough memory\n");
            return -1;
        }
        if ((y[k] = (double *)malloc(SIZE*sizeof(double))) == NULL) {
            printf("Not enough memory\n");
            return -1;
        }
        if ((q[k] = (double *)malloc(SIZE*sizeof(double))) == NULL) {
            printf("Not enough memory\n");
            return -1;
        }
    }


    clock_gettime(CLOCK_MONOTONIC, &start);

    // Initialization
    A[0] = 1.0/L;
    alpha[0] = A[0];
    iterations = 0;
    prev = 0;
    next = 1;

    for (k = 0; k < SIZE; k++) {
        y[0][k] = CONST;
    }

    set_t(t);
    if (gradient_J(y[0], x, t, gradient) < 0) {
        printf("Not enough memory\n");
        return -1;
    }

    for (k = 0; k < SIZE; k++) {
        u[0][k] = y[0][k] - alpha[0]*gradient[k];
        q[0][k] = u[0][k];
    }

    accept_value(u[0]);
    accept_value(q[0]);


    // STM
    do {
        alpha[next] = 1.0/(2*L) + sqrt(1.0/(4*L*L) + A[prev]/L);
        A[next] = A[prev] + alpha[next];

        for (k = 0; k < SIZE; k++) {
            y[next][k] = (alpha[next]*u[prev][k] + A[prev]*q[prev][k])/A[next];
        }
        if (gradient_J(y[next], x, t, gradient) < 0) {
            printf("Not enough memory\n");
            return -1;
        }

        for (k = 0; k < SIZE; k++) {
            u[next][k] = u[prev][k] - alpha[next]*gradient[k];
            q[next][k] = (alpha[next]*u[next][k]+A[prev]*q[prev][k])/A[next];
        }

        accept_value(u[next]);
        accept_value(q[next]);

        //J_prev = J(q[prev], x, t);
        J_next = J(q[next], x, t);
        iterations++;

        k = prev;
        prev = next;
        next = k;
    } while (!STOPPING_RULE(J_next));

    clock_gettime(CLOCK_MONOTONIC, &end);

    for (k = 0; k < SIZE; k++) {
        printf("(%lf,%lf) ", t[k], q[prev][k]);
    }

    printf("\nIterations: %d, J = %lf\n", iterations, J_next);
    printf("Time: %lf s \n", (double)((end.tv_sec - start.tv_sec)*1000000000L + (end.tv_nsec - start.tv_nsec))/1000000000L );

    free(t);
    free(x);
    free(gradient);
    for (k = 0; k < 2; k++) {
        free(u[k]);
        free(y[k]);
        free(q[k]);
    }

    return 0;
}
