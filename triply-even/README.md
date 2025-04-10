# On some $K=1$ Codes Admitting Strongly Transversal T

Here you can find the parity check matrices for the following codes admitting strongly transversal T gates, i.e., physical $T$ gates implements a logical $T$ or $T^\dagger$ gate (depending on blocklength $N\equiv 1$ or $-1$ mod $8$).

They are obtained via [doubling transform](http://dx.doi.org/10.1103/PhysRevA.86.052329) on a triply-even (TE) quantum CSS code, i.e., all $X$ stabilizers have weight divisible by $8$ and a self-dual doubly-even (DE) CSS code, i.e., $X$ and $Z$ stabilizers are the same and weight divisible by $4$. All the codes in this repo are of odd blocklength, and having all-one as both $X$ and $Z$ logical operators.

* [[[49,1,5]]](http://dx.doi.org/10.1103/PhysRevA.86.052329) from the TE [[15,1,3]] Reed-Muller code and DE [[17,1,5]] color code
* [[[95,1,7]]](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.042416) from TE 49 code and DE Golay [[23,1,7]] code (which is also QR)
* $[[177,1,9]]$ from TE 95 code and DE [[41,1,9]] code I found
* $[[271,1,11]]$ from TE 177 code and DE [[47,1,11]] QR code
* $[[401,1,13]]$ from TE 271 code and DE [[65,1,13]] code I found
* $[[559,1,15]]$ from TE 401 code and DE [[79,1,15]] QR code

The usage of QR codes are proposed by ["Transversal Clifford and T-gate codes of short length and high distance"](https://arxiv.org/abs/2408.12752). Matrices of QR code are extraced from MAGMA, as described in `matrix_processing.ipynb`. This notebook also contains useful scripts in doing doubling transform, read and write matrix in human-readable format or MAGMA format or MTX format.

I will upload a short note on how the two DE codes [[41,1,9]] and $[[65,1,13]]$ are found.
## Distance verification

To verify the distance of the triply-even codes in GAP using the [QDistRnd](https://github.com/QEC-pages/QDistRnd) package: 
- copy all `*.mtx` files to `{PATH to GAP}/pkg/qdistrnd/matrices/`
- then invoke GAP and enter the following (the same can be done by replacing 49 with 95, 177, 271, 401, 559)
```
gap> LoadPackage("QDistRnd");
gap> filedir:=DirectoriesPackageLibrary("QDistRnd","matrices");;
gap> lisX:=ReadMTXE(Filename(filedir,"49_X.mtx"),0);; GX:=lisX[3];;
gap> lisZ:=ReadMTXE(Filename(filedir,"49_Z.mtx"),0);; GZ:=lisX[3];;
gap> DistRandCSS(GX,GZ,1000,1,0:field:=GF(2));
```

For doubly-even codes (n=41, 65), the following suffices
```
gap> lisX:=ReadMTXE(Filename(filedir,"41_XZ.mtx"),0);; GX:=lisX[3];;
gap> DistRandCSS(GX,GX,1000,1,0:field:=GF(2));
```