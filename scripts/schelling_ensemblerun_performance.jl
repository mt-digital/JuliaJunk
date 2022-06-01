using Distributed
using Printf

# Separate activation statement from @everywhere using statements to make sure 
# environment activated across all processors before `using` dependencies.
@everywhere begin
    using DrWatson; quickactivate("..")
end

@everywhere begin
    using Agents
    using Agents.Models: schelling, schelling_agent_step!
end

# Precompile across processors to ensure no lag on first model runs.
@everywhere begin
    using Pkg
    Pkg.precompile()
end


function test_schelling_ensemble_versions(
        versions = [:serial, :old, :darray];
        nreplicates = 10, numagents_low = 200, numagents_high=300, nsteps=100
    )

    results = Dict(k => 0.0 for k in versions)

    for version in versions

        basemodels = [Models.schelling(; numagents)[1] 
                      for numagents in numagents_low:numagents_high]

        models = repeat(basemodels, nreplicates)

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
        @printf "%s finished in %.2fs\n" version results[version]
    end

    return results
end


function ensemblerun_benchmark(nreplicates, nsteps, versions=[:serial, :current, :darray])
    println("ENSEMBLE RUN BENCHMARK, nreplicates=$nreplicates, nsteps=$nsteps")
    result = test_schelling_ensemble_versions(
        [:serial, :current, :darray]; nreplicates=nreplicates, nsteps=nsteps
    )
    @printf "Parallel-to-serial ratio (current): %.2f\n" result[:current]/result[:serial]
    @printf "Parallel-to-serial ratio (darray version): %.2f\n\n" result[:darray]/result[:serial]
end


function main()

    ensemblerun_benchmark(2, 100)
    ensemblerun_benchmark(4, 100)
    ensemblerun_benchmark(10, 2000)
    ensemblerun_benchmark(20, 2000)
    ensemblerun_benchmark(100, 50)

end


main()
