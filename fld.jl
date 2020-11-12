#This file has the definitions for the gray Flux-Limited Diffusion Solver
#The code uses units of
# cm for length
# sh for time (1 sh = 10^-8 s)
# GJ (gigajoules) for energy; 1 GJ is also known as a jerk
# keV for temperature (1 keV is 1.1605 * 10^7 K)

using PyCall
using JLD, HDF5, SparseArrays

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

    redges = range(0,stop=Lr,length=Nr+1)
    zedges = range(0,stop=Lz,length=Nz+1)


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
            A[here,here] = sigma[i,j]
            #add in i+1
            if (i< Nr)
                Diplus = 2*Dright[i,j]*Dleft[i+1,j]/(Dright[i,j] + Dleft[i+1,j])
                tmp_zone = xy_to_vect(i+1,j,Nr,Nz)
                A[here,tmp_zone] = -Siplus*iVij*idr*Diplus
                A[here,here] += Siplus*iVij*idr*Diplus
            elseif (upper_r == "vacuum")
                A[here,here] += Siplus*iVij*idr*Dright[i,j] #*idr*D[i,j]
            elseif (isa(upper_r,Float64))
                
                
                A[here,here] += Siplus*iVij*idr*Dright[i,j] #Siplus*iVij*0.5*c #*idr*D[i,j]
                b[here] += Siplus*iVij*idr*Dright[i,j]*upper_r
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
                A[here,here] += idz2*Dtop[i,j]   #idz2*D[i,j]
            elseif (isa(upper_z,Float64))
                A[here,here] += idz2*Dtop[i,j] #idz*0.5*c #idz2*D[i,j] #
                b[here] += idz2*Dtop[i,j]*upper_z # idz*upper_z*0.5*c #
            end
            #add in j-1
            if (j>1)
                Djminus = 2*Dbottom[i,j]*Dtop[i,j-1]/(Dbottom[i,j] + Dtop[i,j-1])
                tmp_zone = xy_to_vect(i,j-1,Nr,Nz)
                A[here,tmp_zone] = -idz2*Djminus
                A[here,here] += idz2*Djminus
            elseif (lower_z == "vacuum")
                A[here,here] += idz2*Dbottom[i,j] #idz*0.5*c #idz2*D[i,j]
            elseif (isa(lower_z,Float64))
                A[here,here] += idz2*Dbottom[i,j] #idz*c*0.5 #
                b[here] += idz2*Dbottom[i,j]*lower_z #idz*lower_z*c #
            end
        end #for j
    end #for i
    A,b
end #function

function time_dep_RT(Tfinal,delta_t,T,Er,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,
Nr,Nz,Lr,Lz;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld")
    #sigma,Q,Cv,EOS,invEOS are functions of t,T,Nr,Nz,Lr,Lz
    #D is function of t,T,Nr,Nz,,Lr,Lz,Er,sigma
    done = false
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
        sigma = sigma_func(t,T,Nr,Nz,Lr,Lz)
        Cv = Cv_func(t,T,Nr,Nz,Lr,Lz)
        Q = Q_func(t,T,Nr,Nz,Lr,Lz)
        beta = 4*a*c*reshape(T,Nr,Nz).^3 ./ Cv
        f = 1 ./(1 .+ beta.*sigma*dt)
        sigma_a = sigma.*f
        sigma_star = c*sigma_a .+ 1.0/(dt)
        Trect = reshape(T,Nr,Nz)

        #define Tleft
        tmp = ones(Nr,Nz)
        tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dleft = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
        #define Tright
        tmp = ones(Nr,Nz)
        tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dright = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
        #define Ttop
        tmp = ones(Nr,Nz)
        tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dtop = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
        #define Tbottom
        tmp = ones(Nr,Nz)
        tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dbottom = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)

        A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_star, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
        b += Er/(dt) + reshape(Q,(Nr*Nz)) + c*a*T.^4 .*reshape(sigma_a,Nr*Nz)
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

        #save(fname, "Nr", Nr, "Nz", Nz, "dr", dr, "dz", dz, "times", times, "T", T_t, "Er", Er_t)
    end #timestep loop
    if (LOUD == -1)
        println("Step $(step-1), t = $(time)")
    end
    times,Er_t,T_t
end #function

function time_dep_3T(Tfinal,delta_t,Te,Ti,Er,D_func,sigma_func,Q_func,Qe_func,Qi_func,Cve_func,EOSe,invEOSe,Cvi_func,EOSi,invEOSi,gamma_func,
Nr,Nz,Lr,Lz;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld")
    #sigma,Q,Cv,EOS,invEOS are functions of t,T,Nr,Nz,Lr,Lz
    #D is function of t,T,Nr,Nz,,Lr,Lz,Er,sigma
    done = false
    time = 0
    dz = Lz/Nz
    dr = Lr/Nr
    times = [0.]
    steps = Int64(ceil(Tfinal/delta_t+1))
    println(steps)
    Er_t = zeros(Nz*Nr,steps+1)
    Er_t[:,1] = Er
    Te_t = zeros(Nz*Nr,steps+1)
    Te_t[:,1] = Te
    Ti_t = zeros(Nz*Nr,steps+1)
    Ti_t[:,1] = Ti
    step = 1
    while !(done)
        dt = min(Tfinal-time, delta_t)
        t = time+dt
        sigma = sigma_func(t,Te,Nr,Nz,Lr,Lz)
        Cve = Cve_func(t,Te,Nr,Nz,Lr,Lz)
        Cvi = Cvi_func(t,Ti,Nr,Nz,Lr,Lz)
        Q = Q_func(t,Te,Nr,Nz,Lr,Lz)
        Qe = Qe_func(t,Te,Nr,Nz,Lr,Lz)
        Qi = Qi_func(t,Ti,Nr,Nz,Lr,Lz)
        gamma = gamma_func(t,Te,Ti,Nr,Nz,Lr,Lz)
        g_res = copy(reshape(gamma,Nr,Nz))
        
        Ti_rect = reshape(Ti,Nr,Nz)
        Trect = reshape(Te,Nr,Nz)
        g_EvDens = Cvi./dt .+ g_res
        Si = g_res./g_EvDens.*(Cvi./dt.*Ti_rect + Qi)
        alpha = g_res.*(g_res./g_EvDens.-1)
        Sn = alpha.*Trect + Si + Qe
        
        
        beta = 4*a*c*reshape(Te,Nr,Nz).^3 ./ Cve
        f = (Cve-alpha*dt)./((1 .+ beta.*sigma*dt).*Cve-alpha*dt) #1 ./(1 .+ beta.*sigma*dt)
        sigma_a = sigma.*f
        sigma_star = c*sigma_a .+ 1.0/(dt)

        #define Tleft
        tmp = ones(Nr,Nz)
        tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dleft = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
        #define Tright
        tmp = ones(Nr,Nz)
        tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dright = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
        #define Ttop
        tmp = ones(Nr,Nz)
        tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dtop = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
        #define Tbottom
        tmp = ones(Nr,Nz)
        tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dbottom = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)

        A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_star, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
        b += Er/(dt) + reshape(Q,(Nr*Nz)) + c*a*Te.^4 .*reshape(sigma_a,Nr*Nz) + reshape((1.0 .-f).*Sn,Nr*Nz)
        Er = A\b

        dE = EOSe(t,Te,Nr,Nz,Lr,Lz) + (reshape(f.*Sn,Nr*Nz)+c*reshape(sigma_a,Nr*Nz).*(Er - a*Te.^4)).*reshape((Cve./(Cve-alpha*dt)),Nr*Nz)*dt
        
        dEi = EOSi(t,Ti,Nr,Nz,Lr,Lz) - dt*reshape(Sn,Nr*Nz) - dt*(reshape(f.*Sn,Nr*Nz)+c*reshape(sigma_a,Nr*Nz).*(Er - a*Te.^4)).*reshape((alpha*dt./(Cve-alpha*dt)),Nr*Nz) + dt*reshape(Qe+Qi,Nr*Nz)
        Te = copy(invEOSe(t,dE,Nr,Nz,Lr,Lz))
        Ti = copy(invEOSi(t,dEi,Nr,Nz,Lr,Lz))
        
        time += dt
        done = time >= Tfinal
        times = push!(times,time)
        if (LOUD >0)
            println("Step $(step), t = $(time)")
        end
        step += 1
        Er_t[:,step] = copy(Er)
        Te_t[:,step] = copy(Te)
        Ti_t[:,step] = copy(Ti)

        #save(fname, "Nr", Nr, "Nz", Nz, "dr", dr, "dz", dz, "times", times, "Te", Te_t, "Ti", Ti_t, "Er", Er_t)
    end #timestep loop
    if (LOUD == -1)
        println("Step $(step-1), t = $(time)")
    end
    times,Er_t,Te_t, Ti_t
end #function

function time_dep_RT_one_it(Tfinal,delta_t,T,Er,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,
Nr,Nz,Lr,Lz;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld")
    #sigma,Q,Cv,EOS,invEOS are functions of t,T,Nr,Nz,Lr,Lz
    #D is function of t,T,Nr,Nz,,Lr,Lz,Er,sigma
    done = false
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
        sigma = sigma_func(t,T,Nr,Nz,Lr,Lz)
        Cv = Cv_func(t,T,Nr,Nz,Lr,Lz)
        Q = Q_func(t,T,Nr,Nz,Lr,Lz)
        beta = 4*a*c*reshape(T,Nr,Nz).^3 ./Cv
        f = 1. /(1+beta.*sigma*dt)
        sigma_a = sigma.*f
        sigma_star = c*sigma_a + 1.0/(dt)
        Trect = reshape(T,Nr,Nz)

        #define Tleft
        tmp = ones(Nr,Nz)
        tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dleft = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
        #define Tright
        tmp = ones(Nr,Nz)
        tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
        sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dright = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
        #define Ttop
        tmp = ones(Nr,Nz)
        tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dtop = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
        #define Tbottom
        tmp = ones(Nr,Nz)
        tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
        sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
        Dbottom = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)

        A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_star, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
        b += Er/(dt) + reshape(Q,(Nr*Nz)) + c*a*T.^4 .*reshape(sigma_a,Nr*Nz)
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
    times,Er_t,T_t
end #function


function gray_update(t,delta_t,T,Er,Er_prev_it,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,rho,
Nr,Nz,Lr,Lz;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld", use_fact = 1.0)
    sigma = sigma_func(t,T,Nr,Nz,Lr,Lz)
    Cv = Cv_func(t,T,Nr,Nz,Lr,Lz)
    Q = Q_func(t,T,Nr,Nz,Lr,Lz)
    JEe = -c./Cv.*sigma*4*a.*reshape(T,Nr,Nz).^3
    Jee = rho/delta_t + c./Cv.*sigma*4*a.*reshape(T,Nr,Nz).^3
    sigma_a = sigma.*(1+use_fact*JEe./Jee)
    println((JEe./Jee))
    sigma_star = c*sigma_a + 1.0/(delta_t)
    Trect = reshape(T,Nr,Nz)

    #define Tleft
    tmp = ones(Nr,Nz)
    tmp[2:Nr,:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
    sigma_left = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
    Dleft = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_left)
    #define Tright
    tmp = ones(Nr,Nz)
    tmp[1:(Nr-1),:] = 0.5*(Trect[1:(Nr-1),:] + Trect[2:Nr,:])
    sigma_right = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
    Dright = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_right)
    #define Ttop
    tmp = ones(Nr,Nz)
    tmp[:,1:(Nz-1)] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
    sigma_top = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
    Dtop = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_top)
    #define Tbottom
    tmp = ones(Nr,Nz)
    tmp[:,2:Nz] = 0.5*(Trect[:,1:(Nz-1)] + Trect[:,2:Nz])
    sigma_bottom = sigma_func(t,tmp,Nr,Nz,Lr,Lz)
    Dbottom = D_func(t,tmp,Nr,Nz,Lr,Lz,Er,sigma_bottom)

    A,b = create_A(c*Dleft,c*Dright, c*Dtop,c*Dbottom, sigma_star, Nr, Nz, Lr, Lz, lower_z=lower_z,upper_z=upper_z, upper_r=upper_r);
    b += Er/(dt) + reshape(Q,(Nr*Nz)) + c*a*reshape(sigma_a,Nr*Nz).*(T.^4 + use_fact*reshape(JEe./Jee,Nr*Nz).*Er_prev_it)
    #println(b,T,sigma,Q,Er/dt)
    Er = A\b
    
    return Er
    end #function

function gray_GARNET(Tfinal,delta_t,T,Er,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,single_invEOS,rho,
Nr,Nz,Lr,Lz;lower_z = "refl",upper_z="refl",upper_r="refl", LOUD=0, fname="tmp.jld", tol_e = 1e-6, tol_Er = 1e-6, max_its = 20, conserve = true)
    #sigma,Q,Cv,EOS,invEOS are functions of t,T,Nr,Nz,Lr,Lz
    #D is function of t,T,Nr,Nz,,Lr,Lz,Er,sigma
    done = false
    time = 0
    dz = Lz/Nz
    dr = Lr/Nr
    times = [0.]
    steps = Int64(round(Tfinal/delta_t+1))
    println(steps)
    Er_t = zeros(Nz*Nr,steps)
    Er_t[:,1] = Er
    T_t = zeros(Nz*Nr,steps)
    diff_solves = zeros(steps)
    outer_its = zeros(steps)
    T_t[:,1] = T
    step = 1
    eta = zeros(Nr,Nz)
    del_i = 1e-6*1im
    it = 1
    while !(done)
        dt = min(Tfinal-time, delta_t)
        t = time+dt
        e_old = EOS(t,T_t[:,step],Nr,Nz,Lr,Lz)
        e = copy(e_old)
        sigma = sigma_func(t,T,Nr,Nz,Lr,Lz)
        #initialize temperature
        for i in 1:Nr
            for j in 1:Nz
                l = xy_to_vect(i,j,Nr,Nz)
                f(e) = rho[i,j]/dt*(e-e_old[l]) + c*sigma[i,j]*(a*single_invEOS(t,e,i,j,Nr,Nz,Lr,Lz)^4 - Er[l])
                fp(e) = imag(f(e+del_i)/imag(del_i))
                e[l],its = nm(f, fp, e[l]; tol=sqrt(eps()))
                
                ft(T) = a*T^4 + EOS(t,T,i,j,Nr,Nz,Lr,Lz) - (e_old[l] + Er[l])
                ftp(T) = imag(ft(T+del_i)/imag(del_i))
                Teq, its = nm(ft, ftp, 0.5*(T[l]+ (Er[l]/a)^.25); tol=sqrt(eps()))
                e_eq = EOS(t,Teq,i,j,Nr,Nz,Lr,Lz)
                Er_eq = a*Teq^4
                
                println("Er_prev = ",Er[l], " Er_eq = ", Er_eq, " T_prev = ", T_t[l,step], " T_init = ", 
                    single_invEOS(t,e[l],i,j,Nr,Nz,Lr,Lz), " T_eq = ", Teq)
                omega = 1.0/(1+c*sigma[i,j]*dt)
                e[l] = e[l] * omega + e_eq*(1-omega)
                Er[l] = Er[l]*omega + Er_eq*(1-omega)
                
                
            end #for j
        end #for i
        T = copy(invEOS(t,e,Nr,Nz,Lr,Lz))
        e_oi = copy(e)
        #println("e = ",e)
        #println("T = ",T)
        converged = false
        Er_o = copy(Er_t[:,step])
        Er_oi = copy(Er_o)
        while !(converged)
            conv_E = false
            E_its = 1
            sigma = sigma_func(t,T,Nr,Nz,Lr,Lz)
            Cv = Cv_func(t,T,Nr,Nz,Lr,Lz)
            Er = gray_update(t,dt,T,Er_t[:,step],Er_oi,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,rho,
                             Nr,Nz,Lr,Lz;lower_z=lower_z,upper_z=upper_z,upper_r=upper_r, LOUD=LOUD, fname=fname)
            diff_solves[step] += 1
            #solve for new temperature in each zone
            for i in 1:Nr
                for j in 1:Nz
                    l = xy_to_vect(i,j,Nr,Nz)
                    f(e) = rho[i,j]/dt*(e-e_old[l]) + c*sigma[i,j]*(a*single_invEOS(t,e,i,j,Nr,Nz,Lr,Lz)^4 - Er[l])
                    fp(e) = imag(f(e+del_i)/imag(del_i))
                    #println(fp(e[l])," ",(f(e[l]+1e-6) - f(e[l]))/1.0e-6)
                    e[l],its = nm(f, fp, e[l]; tol=sqrt(eps()))
                end #for j
            end #for i
            T = copy(invEOS(t,e,Nr,Nz,Lr,Lz))
            
            #println("e = ",e)
            #println("Er = ",Er)
            #println("T = ",T)
            converged = ( (norm(e - e_oi) < tol_e*norm(e)) & (norm(Er - Er_oi) < tol_Er*norm(Er))) | (it >= max_its)
            #println(norm(e - e_oi)/norm(e) , norm(Er - Er_oi)/norm(Er))
            Er_oi = copy(Er)
            Er_o = copy(Er)
            e_oi = copy(e)
            it += 1
            outer_its[step] += 1
        end
        
        if (conserve == true)
        #do one more solve and energy update
            Er = gray_update(t,dt,T,Er_t[:,step],0*Er,D_func,sigma_func,Q_func,Cv_func,EOS,invEOS,rho,
                                 Nr,Nz,Lr,Lz;lower_z=lower_z,upper_z=upper_z,upper_r=upper_r, LOUD=LOUD, fname=fname, use_fact = 0.0)
            diff_solves[step] += 1
            #update e in each zone
            for i in 1:Nr
                for j in 1:Nz
                    l = xy_to_vect(i,j,Nr,Nz)
                    e[l]= e_old[l] - dt/rho[i,j]*(c*sigma[i,j]*(a*T[l]^4 - Er[l]))
                end #for j
            end #for i
            T = copy(invEOS(t,e,Nr,Nz,Lr,Lz))
            
        end
        
        time += dt
        done = time >= Tfinal
        times = push!(times,time)
        if (LOUD >0)
            println("Step $(step), t = $(time)")
            println("Energy = ", sum(e+Er))
        end
        step += 1
        Er_t[:,step] = copy(Er)
        T_t[:,step] = copy(T)

        save(fname, "Nr", Nr, "Nz", Nz, "dr", dr, "dz", dz, "times", times, "T", T_t, "Er", Er_t)
    end #timestep loop
    if (LOUD == -1)
        println("Step $(step-1), t = $(time)")
    end
    times,Er_t,T_t, diff_solves, outer_its
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

