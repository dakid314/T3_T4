# install.packages("protr")
# library("protr")


generate_function <- function(path_to_fasta_, path_to_out) {
    x <- protr::readFASTA(
        path_to_fasta_
    )

    result <- Reduce(function(x, y) rbind(x, y), list(
        sapply(x, function(x) protr::extractAAC(x)),
        sapply(x, function(x) protr::extractDC(x)),
        sapply(x, function(x) protr::extractQSO(x)),
        sapply(x, function(x) protr::extractCTDC(x)),
        sapply(x, function(x) protr::extractCTDT(x))
    ))


    write.csv(t(result), path_to_out)
}

# T1
generate_function(
    "tmp/Bigger30/T1/t_n.bigger30.fasta",
    "tmp/T1.t_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T1/t_p.bigger30.fasta",
    "tmp/T1.t_p.protr.csv"
)
generate_function(
    "tmp/Bigger30/T1/v_n.bigger30.fasta",
    "tmp/T1.v_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T1/v_p.bigger30.fasta",
    "tmp/T1.v_p.protr.csv"
)

# T2
generate_function(
    "tmp/Bigger30/T2/t_n.bigger30.fasta",
    "tmp/T2.t_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T2/t_p.bigger30.fasta",
    "tmp/T2.t_p.protr.csv"
)
generate_function(
    "tmp/Bigger30/T2/v_n.bigger30.fasta",
    "tmp/T2.v_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T2/v_p.bigger30.fasta",
    "tmp/T2.v_p.protr.csv"
)

# T3
generate_function(
    "tmp/Bigger30/T3/t_n.bigger30.fasta",
    "tmp/T3.t_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T3/t_p.bigger30.fasta",
    "tmp/T3.t_p.protr.csv"
)
generate_function(
    "tmp/Bigger30/T3/v_n.bigger30.fasta",
    "tmp/T3.v_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T3/v_p.bigger30.fasta",
    "tmp/T3.v_p.protr.csv"
)

# T4
generate_function(
    "tmp/Bigger30/T4/t_n.bigger30.fasta",
    "tmp/T4.t_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T4/t_p.bigger30.fasta",
    "tmp/T4.t_p.protr.csv"
)
generate_function(
    "tmp/Bigger30/T4/v_n.bigger30.fasta",
    "tmp/T4.v_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T4/v_p.bigger30.fasta",
    "tmp/T4.v_p.protr.csv"
)

# T6
generate_function(
    "tmp/Bigger30/T6/t_n.bigger30.fasta",
    "tmp/T6.t_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T6/t_p.bigger30.fasta",
    "tmp/T6.t_p.protr.csv"
)
generate_function(
    "tmp/Bigger30/T6/v_n.bigger30.fasta",
    "tmp/T6.v_n.protr.csv"
)
generate_function(
    "tmp/Bigger30/T6/v_p.bigger30.fasta",
    "tmp/T6.v_p.protr.csv"
)
