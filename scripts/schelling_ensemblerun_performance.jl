using Distributed

@everywhere begin
    using DrWatson; quickactivate("..")
end

@everywhere begin
    using Agents
    using Agents.Models: schelling, schelling_agent_step!
end


function test_schelling_ensemble_versions(
        versions = [:serial, :old, :darray];
        nreplicates = 10, numagents_low = 200, numagents_high=300, nsteps=100
    )

    results = Dict(k => 0.0 for k in versions)

    for version in keys(results)

        basemodels = [Models.schelling(; numagents)[1] 
                      for numagents in numagents_low:numagents_high]

        models = repeat(basemodels, 
                        nreplicates)

        if version == :serial
            results[version] = @elapsed (
                ensemblerun!(models, schelling_agent_step!, dummystep, nsteps;
                             parallel = false)
            )
        else
            results[version] = @elapsed (
                ensemblerun!(models, schelling_agent_step!, dummystep, nsteps;
                             version, parallel = true)
            )
        end
        println("$version finished in $(results[version])s")
    end

    return results
end


function main()

    println("ENSEMBLE RUN TEST, n_replicates=2, nsteps=100")
    result = test_schelling_ensemble_versions(
        [:serial, :current, :darray]; nreplicates=2, nsteps=100
    )
    println("Parallel-to-serial ratio (current): $((result[:current]/result[:serial]))")
    println("Parallel-to-serial ratio (darray version): $(result[:darray]/result[:serial])\n")

    println("ENSEMBLE RUN TEST, n_replicates=4, nsteps=100")
    result = test_schelling_ensemble_versions(
        [:serial, :current, :darray]; nreplicates=4, nsteps=100
    )
    println("Parallel-to-serial ratio (current): $(result[:current]/result[:serial])")
    println("Parallel-to-serial ratio (darray version): $(result[:darray]/result[:serial])\n")
        
    println("ENSEMBLE RUN TEST, n_replicates=10, nsteps=2000")
    result = test_schelling_ensemble_versions(
        [:serial, :darray]; nreplicates=10, nsteps=2000
    )
    println("Parallel-to-serial ratio (darray version): $(result[:darray]/result[:serial])")

end

main()
