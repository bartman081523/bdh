from bdh_gpu_neuro_ref import BDHGPURef, BDHNeuroRef

# Baseline (Def.4)
base = BDHGPURef(n=128, d=24, V=2048, seed=3, u_decay=0.97, ln_before_Dy=True, use_relu_lowrank=True)
print(base.run(T=32))

# Neuro-Variante mit mehreren Prinzipien
neuro = BDHNeuroRef(n=128, d=24, V=2048, seed=3,
                    U_kernels=[0.99, 0.97, 0.94], U_weights=[0.5, 0.3, 0.2],
                    local_forget_eta=0.02, homeostasis_tau=0.15*128,
                    k_wta=16, branches=2, branch_nl="softplus",
                    mod_gamma_max=0.8, spike_rate=0.01,
                    ln_before_Dy=True, use_relu_lowrank=True)
print(neuro.run(T=32))
