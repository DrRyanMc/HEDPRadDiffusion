#This file has the definitions for the gray Flux-Limited Diffusion Solver
#The code uses units of
# cm for length
# sh for time (1 sh = 10^-8 s)
# GJ (gigajoules) for energy; 1 GJ is also known as a jerk
# keV for temperature (1 keV is 1.1605 * 10^7 K)

using PyCall
using JLD, HDF5

#radiation constant in GJ/keV^4/cm^3
const a = 0.01372
#speed of light in cm/sh; 1 sh is 10^-8 s
const c = 299.98
const Planck_int_const = 15*a*c/(4*pi^5)

"This function maps an Nx*Ny 1-D array to a i,j on an Nx by Ny grid"
function vect_to_xy(l,Nx,Ny)
    j = div(l,Nx) + 1
    i = mod(l-j,Nx) + 1
    (i,j)
end

"This function maps i,j on an Nx by Ny grid to an Nx*Ny 1-D array"
function xy_to_vect(i,j,Nx,Ny)
l = Nx*(j-1) + i
l
end
function nm(f, fp, x; tol=sqrt(eps()))
   
    ctr, max_steps = 0, 100
     
    while (abs(f(x)) > tol) && ctr < max_steps
        x = x - f(x) / fp(x)
        ctr = ctr + 1
    end

    ctr >= max_steps ? error("Method did not converge.") : return (x, ctr)
    
end


"This function computes the integral of the planck function from E1 = h*nu [keV] to infinity at T[keV]"
function B1(E1,T)
    r = (E1./T)

    num_terms = convert(Int32,round(minimum([2+20/minimum(r),1024])))
    output = 0*T
    for n in 1:num_terms
        invn = 1.0/n
        output += invn*exp.(-n*r).*(r.^3 + 3*r.^2*invn .+ 6*r * invn^2 .+ 6*invn^3)
    end
    return output*Planck_int_const
end

"This function gives Bg(T)"
function Bg(E1,E2,T)
    return B1(E1,T) - B2(E2,T)
end

"This function computes the derivative of B1 w.r.t. T"
function DB1DT(y,T)
    r = y./T
    num_terms = convert(Int32,round(minimum([2+20/minimum(r),1024])))
    output = 0*T
    for n in 1:num_terms
        output += exp.(-n*r).*(24*T.^4 .+ 24*n*T.^3 .*y + 12*n^2*T.^2 .*y.^2 + 4*n^3*T.*y.^3 .+ n^4*y.^4)./(n^4*T)
    end
    return output*Planck_int_const
end
"This function computes the derivative of Bg w.r.t. T"
function DBgDT(E1,E2,T)
    return DB1DT(E1,T) .- DB1DT(E2,T)
end
"""
    create_A(D,sigma,Nr,Nz,Lr,Lz; lower_z = "refl",upper_z="refl",upper_r="refl")

Compute the matrix and RHS for an RZ diffusion problem
# Arguments
- `D::Float64(Nr,Nz)`: matrix of diffusion coefficients
- `sigma::Float64(Nr,Nz)`: matrix of absorption coefficients
- `Nr::Integer`: number of cells in r
- `Nz::Integer`: number of cells in z
- `Lr::Float64`: size of domain in r
- `Lz::Float64`: size of domain in z
...
"""
function create_A(Dleft,Dright, Dtop,Dbottom, sigma,Nr,Nz,Lr,Lz; lower_z = "refl",upper_z="refl",upper_r="refl")

    redges = linspace(0,Lr,Nr+1)
    zedges = linspace(0,Lz,Nz+1)


    A = spzeros(Nr*Nz,Nr*Nz)
    b = zeros(Nr*Nz)
    dz = Lz/Nz
    dr = Lr/Nr
    idz = 1/dz
    idz2 = idz*idz
    idr = 1/dr

    #loop over cells
    for i in 1:Nr
        for j in 1:Nz
            here = xy_to_vect(i,j,Nr,Nz)
            Vij = pi*(redges[i+1]^2 - redges[i]^2)
            iVij = 1/Vij
            Siplus = 2*pi*redges[i+1]
            Siminus = 2*pi*redges[i]
            #add in diag
            A[here,here] = sigma[i,j]*c
            #add in i+1
            if (i< Nr)
                Diplus = 2*Dright[i,j]*Dleft[i+1,j]/(Dright[i,j] + Dleft[i+1,j])
                tmp_zone = xy_to_vect(i+1,j,Nr,Nz)
                A[here,tmp_zone] = -Siplus*iVij*idr*Diplus
                A[here,here] += Siplus*iVij*idr*Diplus
            elseif (upper_r == "vacuum")
                A[here,here] += Siplus*iVij*0.5*c #*idr*D[i,j]
            elseif (isa(upper_r,Float64))
                A[here,here] += Siplus*iVij*0.5*c #*idr*D[i,j]
                b[here] += Siplus*iVij*0.5*upper_r*c
            end
            #add in i-1
            if (i>1)
                Diminus = 2*Dleft[i,j]*Dright[i-1,j]/(Dleft[i,j] + Dright[i-1,j])
                tmp_zone = xy_to_vect(i-1,j,Nr,Nz)
                A[here,tmp_zone] = -Siminus*iVij*idr*Diminus
                A[here,here] += Siminus*iVij*idr*Diminus
            end
            #add in j+1
            if (j < Nz)
                Djplus = 2*Dtop[i,j]*Dbottom[i,j+1]/(Dtop[i,j] + Dbottom[i,j+1])
                tmp_zone = xy_to_vect(i,j+1,Nr,Nz)
                #println("l = $(tmp_zone) i = $(i), j = $(j+1)")
                A[here,tmp_zone] = -idz2*Djplus
                A[here,here] += idz2*Djplus
            elseif (upper_z == "vacuum")
                A[here,here] += idz*0.5*c   #idz2*D[i,j]
            elseif (isa(upper_z,Float64))
                A[here,here] += idz*0.5*c #idz2*D[i,j]
                b[here] += idz*upper_z*0.5*c #idz2*D[i,j]*upper_z
            end
            #add in j-1
            if (j>1)
                Djminus = 2*Dbottom[i,j]*Dtop[i,j-1]/(Dbottom[i,j] + Dtop[i,j-1])
                tmp_zone = xy_to_vect(i,j-1,Nr,Nz)
                A[here,tmp_zone] = -idz2*Djminus
                A[here,here] += idz2*Djminus
            elseif (lower_z == "vacuum")
                A[here,here] += idz*0.5*c #idz2*D[i,j]
            elseif (isa(lower_z,Float64))
                A[here,here] += idz*c*0.5 #idz2*D[i,j]
                b[here] += idz*lower_z*c #idz2*D[i,j]*lower_z
            end
        end #for j
    end #for i
    A,b
end #function


"This function solves equation 19 from Brunner's paper. It gives E_star_g"
function Estar_g(delta_t,T,Eg,Eg_l,D_func,sigma_func,Q_func,RHS,Nr,Nz,Lr,Lz;
    lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0)
    
    sigma_star = c*sigma_func(T) + 1.0/(dt)
    dz = Lz/Nz
    dr = Lr/Nr
    A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
    b += RHS
    return A\b
end #function


"One step of Equation 17"
function Estep(delta_t,T,T_prev,E,E_prev,E_l,D_func_arr,sigma_func_arr,Q_func_arr,group_bounds,EOS,Inv_EOS,Cv,rho,Nr,Nz,Lr,Lz,G; lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0)
    idt = 1.0/delta_t
    Jee = zeros(Nr*Nz)
    lagged_Emission = 0*Eg
    for g in 1:G
        Jee += sigma_func_arr[g](T).*DBgDT(group_bounds[g],group_bounds[g+1],T)
        lagged_Emission += reshape(sigma_func_arr[g](T),Nr*Nz).*(reshape(Bg(group_bounds[g],group_bounds[g+1],T),Nr*Nz) - E[:,g])*c
    end
    Jee *= c./reshape(Cv(T))
    Jee +=  reshape(rho,Nr,Nz)*idt 
    
    Esteps = zeros(Nr*Nz,G)
    sigma_input = sigma_func_arr[g](T)
    e = EOS(T)
    e_prev = EOS(T_prev)
    for g in 1:G
        RHS = -c./reshape(Cv(T)).*reshape(sigma_func_arr[g](T).*DBgDT(group_bounds[g],group_bounds[g+1],T),Nr,Nz)./Jee
        RHS *= (reshape(rho,Nr,Nz)*idt*(e-e_prev) + reshape(lagged_Emission,Nr,Nz))
        RHS +=  reshape(sigma_func_arr[g](T),Nr,Nz).*reshape(E[:,g],Nr,Nz)*c #this group emission (2nd term on RHS)
        RHS +=  idt*reshape(E[:,g],Nr,Nz) #first term on RHS of 17
        RHS +=  reshape(Q_func(T),Nr,Nz) #source term on RHS of 17
        sigma_input *= (1-c./reshape(Cv(T)).*reshape(sigma_func_arr[g](T).*
                DBgDT(group_bounds[g],group_bounds[g+1],T),Nr,Nz)./Jee)
        sigma_input += idt
        
        #define Tleft
        tmp = ones(Nr,Nz)
        tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dleft = D_func_arr[g](t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
        #define Tright
        tmp = ones(Nr,Nz)
        tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dright = D_func_arr[g](t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
        #define Ttop
        tmp = ones(Nr,Nz)
        tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dtop = D_func_arr[g](t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
        #define Tbottom
        tmp = ones(Nr,Nz)
        tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dbottom = D_func_arr[g](t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)
        
        A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_input, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
        b += RHS
        
        Esteps[:,g] = A\b
        
    end
    
    return Esteps
end #function
