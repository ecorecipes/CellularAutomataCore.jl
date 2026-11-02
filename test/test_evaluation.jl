# Evaluation metrics against hand-computed values.
@testset "evaluation" begin
    p = Bool[1 1 0; 0 1 0]
    o = Bool[1 0 0; 1 1 0]
    @test confusion_counts(p, o) == (tp=2, fp=1, fn=1, tn=2)
    @test_throws ArgumentError confusion_counts(p, trues(3, 2))

    @test cohens_kappa([:a :b; :b :a], [:a :b; :b :a]) == 1.0
    @test cohens_kappa([1 0; 1 0], [1 1; 0 0]) == 0.0
    # p_o = 4/6, p_e = (3·3 + 3·3) / 36 = 1/2, so kappa = (2/3 - 1/2) / (1/2) = 1/3
    @test cohens_kappa(Int.(p), Int.(o)) ≈ 1 / 3
    @test cohens_kappa(fill(:S, 3, 3), fill(:S, 3, 3)) == 1.0
    @test cohens_kappa(fill(:S, 3, 3), fill(:I, 3, 3)) == 0.0
    @test_throws ArgumentError cohens_kappa(Int[], Int[])

    @test auc([0.9, 0.8, 0.3, 0.1], [true, true, false, false]) == 1.0
    @test auc([0.1, 0.3, 0.8, 0.9], [true, true, false, false]) == 0.0
    @test auc([0.5, 0.5, 0.5, 0.5], [true, false, true, false]) == 0.5
    # one discordant pair out of four: 0.75
    @test auc([0.9, 0.2, 0.5, 0.1], [true, true, false, false]) == 0.75
    @test auc(reshape([0.9, 0.2, 0.5, 0.1], 2, 2),
              reshape([true, true, false, false], 2, 2)) == 0.75
    @test_throws ArgumentError auc([0.1, 0.2], [true, true])

    init = [0 0 0 0]
    @test figure_of_merit(init, [1 1 0 0], [1 1 0 0]) == 1.0
    @test figure_of_merit(init, [1 1 0 0], [0 1 1 0]) ≈ 1 / 3
    @test figure_of_merit(init, init, init) == 1.0
    @test figure_of_merit(init, [2 0 0 0], [1 0 0 0]) == 0.0  # changed, but to a different state
end
