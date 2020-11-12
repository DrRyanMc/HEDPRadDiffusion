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
            Vij = pi*(redges[i+1]^2 - redges[i]^2)*dz
            iVij = 1/Vij
            Siplus = 2*pi*redges[i+1]
            Siminus = 2*pi*redges[i]
            #add in diag
            A[here,here] = sigma[i,j]
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
                A[here,here] += idz*c #idz2*D[i,j]
                b[here] += idz*lower_z*c #idz2*D[i,j]*lower_z
            end
        end #for j
    end #for i
    A,b
end #function

function time_dep_RT(Tfinal,delta_t,scale_change,D_func,sigma_func,scale,r_apt,sc_length,Q_func,Cv_func,EOS,invEOS,
Nr,Nz,Lr,Lz,theta;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld")
    #sigma,Q,Cv,EOS,invEOS are functions of t,T,Nr,Nz,Lr,Lz
    #D is function of t,T,Nr,Nz,,Lr,Lz,Er,sigma
    done = false
    Lz,Lr,Nr,Nz,T = scale_change(scale,Lr,Lz,Nr,Nz)
    Er = a*T.^4
    time = 0
    dz = Lz/Nz
    dr = Lr/Nr
    times = [0.]
    steps = Int64(ceil(Tfinal/delta_t+1))
    println(steps)
    Er_t = zeros(Nz*Nr,steps+1)
    Er_t[:,1] = Er
    T_t = zeros(Nz*Nr,steps+1)
    T_t[:,1] = T
    step = 1
    while !(done)
        dt = min(Tfinal-time, delta_t)
        t = time+dt
        sigma = sigma_func(t,T,Nr,Nz,Lr,Lz,scale,r_apt,sc_length,theta)
        Cv = Cv_func(t,T,Nr,Nz,Lr,Lz)
        Q = Q_func(t,T,Nr,Nz,Lr,Lz,scale)
        beta = 4*a*c*reshape(T,Nr,Nz).^3./Cv
        f = 1./(1+beta.*sigma*dt)
        sigma_a = sigma.*f
        sigma_star = c*sigma_a + 1.0/(dt)
        Trect = reshape(T,Nr,Nz)

        #define Tleft
        tmp = ones(Nr,Nz)
        tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz,scale,r_apt,sc_length,theta)
        Dleft = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
        #define Tright
        tmp = ones(Nr,Nz)
        tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz,scale,r_apt,sc_length,theta)
        Dright = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
        #define Ttop
        tmp = ones(Nr,Nz)
        tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz,scale,r_apt,sc_length,theta)
        Dtop = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
        #define Tbottom
        tmp = ones(Nr,Nz)
        tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz,scale,r_apt,sc_length,theta)
        Dbottom = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)

        A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_star, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
        b += Er/(dt) + reshape(Q,(Nr*Nz)) + c*a*T.^4.*reshape(sigma_a,Nr*Nz)
        Er = A\b
        dE = EOS(t,T,Nr,Nz,Lr,Lz) + c*dt*reshape(sigma_a,Nr*Nz).*(Er - a*T.^4)
        T = copy(invEOS(t,dE,Nr,Nz,Lr,Lz))

        time += dt
        done = time >= Tfinal
        times = push!(times,time)
        if (LOUD >0)
            println("Step $(step), t = $(time)")
        end
        step += 1
        Er_t[:,step] = copy(Er)
        T_t[:,step] = copy(T)

        save(fname, "Nr", Nr, "Nz", Nz, "dr", dr, "dz", dz, "times", times, "T", T_t, "Er", Er_t)
    end #timestep loop
    if (LOUD == -1)
        println("Step $(step-1), t = $(time)")
    end
    times,Er_t,T_t,Nr,Nz,Lr,Lz
end #function

function meshgrid(x,y)
    Nx = length(x)
    Ny = length(y)
    X = zeros(Nx,Ny)
    Y = zeros(Nx,Ny)
    for i in 1:Nx
        for j in 1:Ny
            X[i,j] = x[i]
            Y[i,j] = y[j]
        end
    end
    X,Y
end
