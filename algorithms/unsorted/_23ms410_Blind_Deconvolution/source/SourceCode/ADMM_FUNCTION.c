#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Conjugate(double** img_y,double** img_c, double** result,int width,int height, int size_f, int itr){
	double* y;		
	double** c;		
	double** C;
	double** ct; 
	double** M; 
	double* M1;
	double* g0;
	double* d0;
	double* g1;
	double* d1;
	double* f0;		
	double* f;	
	double c1=0.0,c2=0.0,sum=0,alpha,beta;
	int i,j,k,m,n,size,count,a,b,e;
	int Y = height*width;

	size = sqrt(size_f);
	count = 0;
	a = 0;
	b = 0;
	e = 0;
	
	y = (double*)malloc(sizeof(double) * Y);

	for (i = 0; i < height; i++) {
		for(j = 0;j < width; j++){
			y[j + i * height] = img_y[i][j];
		}
	}

	f0 = (double*)malloc(sizeof(double) * Y);
	for (i = 0; i < Y; i++) {
		f0[i] = 0;
	}
	
	C = (double**)malloc(sizeof(double*) * 1000);
	for (i = 0; i < 1000; i++) {
		C[i] = (double*)malloc(sizeof(double) * 1000);
	}

	for (i = 0; i < 1000; i++) {
		for (j = 0; j < 1000; j++) {
			C[i][j] = 0;
		}
	}

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			C[i + size / 2][j + size / 2] = img_c[i][j];
		}
	}
	
	c = (double**)malloc(sizeof(double*) * Y);
	for (i = 0; i < Y; i++) {
		c[i] = (double*)malloc(sizeof(double) * Y);
	}

	for (i = 0; i < Y; i++) {
		for (j = 0; j < Y; j++) {
			c[i][j] = 0;
		}
	}

	for (i = 0; i < Y; i++) {

		for (j = 0; j < size_f; j++) {

			e = j % size;

			c[i][j] = C[count + a][e + b];
			if (j % size == size - 1) {
				count++;
			}
		}

		b++;

		if (i % width == width - 1) {
			a++;
			b = 0;
		}

		count = 0;

	}
	
	for (i = 0; i < 1000; i++) {
		free(C[i]);
	}
	free(C);

	//-------------------------------------------------------------------------

	ct = (double**)malloc(sizeof(double*) * Y);
	for (i = 0; i < Y; i++) {
		ct[i] = (double*)malloc(sizeof(double) * Y);
	}

	M = (double**)malloc(sizeof(double*) * Y);
	for (i = 0; i < Y; i++) {
		M[i] = (double*)malloc(sizeof(double) * Y);
	}

	M1 = (double*)malloc(sizeof(double) * Y);

	g0 = (double*)malloc(sizeof(double) * Y);

	g1 = (double*)malloc(sizeof(double) * Y);

	d0 = (double*)malloc(sizeof(double) * Y);

	d1 = (double*)malloc(sizeof(double) * Y);

	f = (double*)malloc(sizeof(double) * Y);
	

	for (i = 0; i < Y; i++) {
		for (j = 0; j < Y; j++) {
			ct[i][j] = 0;
			M[i][j] = 0;
		}
		M1[i] = 0;
		g0[i] = 0;
		g1[i] = 0;
		d0[i] = 0;
		d1[i] = 0;
		f[i] = 0;
	}
	

	for(i=0;i<Y;i++) {
		for(j=0;j<Y;j++) {
			ct[i][j]=c[j][i];
		}
	}
	
	sum=0;
	for(i=0;i<Y;i++) {
		for(j=0;j<Y;j++) {
			for(n=0;n<Y;n++) {
				sum=sum+ct[i][n]*c[n][j];
			}
		M[i][j]=sum;
		sum=0;
		}
	}
	
	sum=0;
	for(i=0;i<Y;i++) {
		for(j=0;j<Y;j++) {
			sum=sum+ct[i][j]*y[j];
		}
		g0[i]=sum;

		d0[i]=g0[i];
		sum=0;
	}

	for (k = 1; k <= itr; k++) {
		c1 = c2 = 0;
		for (m = 0; m < Y; m++) {
			for (n = 0; n < Y; n++) {
				c1 = c1 + M[m][n] * d0[n];
			}
			M1[m] = c1;
			c2 = c2 + d0[m] * g0[m];
			c1 = 0;
		}

		c1 = 0;
		for (m = 0; m < Y; m++) {
			c1 = c1 + d0[m] * M1[m];
		}

		alpha = c2 / c1;

		for (j = 0; j < Y; j++) {
			f[j] = f0[j] + alpha * d0[j];
		}

		for (m = 0; m < Y; m++) {
			g1[m] = g0[m] - alpha * M1[m];
		}

		c1 = 0;
		for (m = 0; m < Y; m++) {
			c1 = c1 + g1[m] * g1[m];
		}

		beta = c1 / c2;

		for (m = 0; m < Y; m++) {
			d1[m] = g1[m] + beta * d0[m];
			//fprintf(fp,"%f\n",d1[m]);
		}

		for (m = 0; m < Y; m++) {
			f0[m] = f[m];
			g0[m] = g1[m];
			d0[m] = d1[m];
		}

	}
	
	for(i=0;i<size;i++){
		for(j=0;j<size;j++){
			result[i][j] = f[size - j - 1 + size*(size - i -1)];
		}
	}
	
	free(y);
	for (i = 0; i < Y; i++) {
		free(c[i]);
	}
	free(c);
	for (i = 0; i < Y; i++) {
		free(ct[i]);
	}
	free(ct);
	for (i = 0; i < Y; i++) {
		free(M[i]);
	}
	free(M);
	free(M1);
	free(g0);
	free(g1);
	free(d0);
	free(d1);
	free(f0);
	free(f);
}

void conv2d(double** gxy, double** fxy, double** hxy, int nx, int ny, int S)
{
	int   i, j, m, n, a, b;
	double** f;
	double** g;
	double** h;
	double** C;

	a = 0;
	b = 0;

	C = (double**)malloc(sizeof(double*) * 1000);
	for (i = 0; i < 1000; i++) {
		C[i] = (double*)malloc(sizeof(double) * 1000);
	}

	for (i = 0; i < 1000; i++) {
		for (j = 0; j < 1000; j++) {
			C[i][j] = 0;
		}
	}

	f = (double**)malloc(sizeof(double*) * nx);
	for (i = 0; i < nx; i++) {
		f[i] = (double*)malloc(sizeof(double) * ny);
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			f[i][j] = fxy[i][j];
		}
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			C[i + S / 2][j + S / 2] = f[i][j];
		}
	}

	h = (double**)malloc(sizeof(double*) * S);
	for (i = 0; i < S; i++) {
		h[i] = (double*)malloc(sizeof(double) * S);
	}

	for (i = 0; i < S; i++) {
		for (j = 0; j < S; j++) {
			h[S - i - 1][S - j - 1] = hxy[i][j];
		}
	}

	g = (double**)malloc(sizeof(double*) * nx);
	for (i = 0; i < nx; i++) {
		g[i] = (double*)malloc(sizeof(double) * ny);
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			g[i][j] = 0;
		}
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			for (m = 0; m < S; m++) {
				for (n = 0; n < S; n++) {
					g[i][j] += C[m + a][n + b] * h[m][n];
				}
			}
			b++;

			gxy[i][j] = g[i][j];
		}
		a++;
		b = 0;
	}
	
	for (i = 0; i < 1000; i++) {
		free(C[i]);
	}
	free(C);
	for (i = 0; i < nx; i++) {
		free(f[i]);
	}
	free(f);
	for (i = 0; i < S; i++) {
		free(h[i]);
	}
	free(h);
	for (i = 0; i < nx; i++) {
		free(g[i]);
	}
	free(g);
}

void WeightFunction(double** img1, double** img2, int nx, int ny ,int Z)
{
	int   i,j;
	double l;

	for (i = 0; i < ny ; i++) {
		for (j = 0; j < nx ; j++) {
			if ((i + 1) == ny && (j + 1) != nx) {
				l = sqrt((img1[i][j + 1] - img1[i][j]) * (img1[i][j + 1] - img1[i][j]) + img1[i][j] * img1[i][j]);
				img2[i][j] = exp(-(l * l) / (2.0 * (double)Z * (double)Z));
			}
			else if ((j + 1) == nx && (i + 1) != ny) {
				l = sqrt((img1[i + 1][j] - img1[i][j]) * (img1[i + 1][j] - img1[i][j]) + img1[i][j] * img1[i][j]);
				img2[i][j] = exp(-(l * l) / (2.0 * (double)Z * (double)Z));
			}
			else if ((i + 1) == ny && (j + 1) == nx) {
				l = sqrt(img1[i][j] * img1[i][j] + img1[i][j] * img1[i][j]);
				img2[i][j] = exp(-(l * l) / (2.0 * (double)Z * (double)Z));
			}
			else {
				l = sqrt((img1[i + 1][j] - img1[i][j]) * (img1[i + 1][j] - img1[i][j]) + (img1[i][j + 1] - img1[i][j]) * (img1[i][j + 1] - img1[i][j]));
				img2[i][j] = exp(-(l * l) / (2.0 * (double)Z * (double)Z));
			}
		}
	}
}

void psf_map(double** psf, double** map, int len_m, int len_n, int len_psf, int height, int width)
{
	int i, j, i0, j0;
	double** r_psf;
	int start_i, start_j;

	for (i = 0; i < len_n; i++) {
		for (j = 0; j < len_m; j++) {
			map[i][j] = 0;
		}
	}

	i0 = 0;
	j0 = 0;
	start_i = 0;
	start_j = 0;

	r_psf = (double**)malloc(sizeof(double*) * len_psf);
	for (i = 0; i < len_psf; i++) {
		r_psf[i] = (double*)malloc(sizeof(double) * len_psf);
	}

	for (i = 0; i < len_psf; i++) {
		for (j = 0; j < len_psf; j++) {
			r_psf[i][j] = psf[len_psf - i - 1][len_psf - j - 1];
		}
	}

	for (i = 0; i < len_n; i++) {

		
		i0 = 0;
		

		for (j = 0; j < len_m; j++) {

			if (j % width == 0) {
				j0 = 0;
			}

			if (i / width < (len_psf / 2)) {

				if (i % width < (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else if (i % width >= width - (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_j + j0 >= 0) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else {
					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_j + j0 >= 0 && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
			}
			else if (i / width >= height - (len_psf / 2)) {

				if (i % width < (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 >= 0 && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else if (i % width >= width - (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 >= 0 && start_j + j0 >= 0) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else {
					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 >= 0 && start_j + j0 >= 0 && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
			}
			else {
				if (i % width < (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_i + i0 >= 0 && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else if (i % width >= width - (len_psf / 2)) {

					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_i + i0 >= 0 && start_j + j0 >= 0) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
				else {
					start_i = (len_psf / 2) - (i / width);

					start_j = (len_psf / 2) - (i % width);

					if (start_i + i0 < len_psf && start_i + i0 >= 0 && start_j + j0 >= 0 && start_j + j0 < len_psf) {
						map[i][j] = r_psf[start_i + i0][start_j + j0];
					}
					else {
						map[i][j] = 0.0;
					}

					j0++;

					if (j % width == width - 1) {
						i0++;
					}
				}
			}
		}
	}
	
	for (i = 0; i < len_psf; i++) {
		free(r_psf[i]);
	}
	free(r_psf);
	
}
