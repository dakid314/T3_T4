library(rlang)
library(ggseqlogo)
library(Biostrings)
library(gridExtra)
library(ggplot2)

install.packages("ggseqlogo")
install.packages("stringr")
install.packages("ellipsis")
install.packages("rlang")

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install()
BiocManager::install("Biostrings")

get_fasta_db <- function(ty, pn, nc) {
    return(unlist(base::lapply(as.character(Biostrings::readBStringSet(
        stringr::str_interp("data/db/T${ty}/${pn}.fasta", ),
        format = "fasta",
        nrec = -1L,
        skip = 0L,
        seek.first.rec = FALSE,
        use.names = FALSE
    )), function(seq_str) {
        if (nc == "n") {
            if (nchar(seq_str) <= 101) {
                return(paste(
                    c(
                        substr(seq_str, 2, 101),
                        paste(
                            replicate(101 - nchar(seq_str), "A"),
                            collapse = ""
                        )
                    ),
                    collapse = ""
                ))
            } else {
                return(substr(seq_str, 2, 101))
            }
        } else if (nc == "c") {
            if (nchar(seq_str) <= 100) {
                return(paste(
                    c(
                        paste(
                            replicate(100 - nchar(seq_str), "A"),
                            collapse = ""
                        ),
                        seq_str
                    ),
                    collapse = ""
                ))
            } else {
                return(substr(seq_str, nchar(seq_str) - 99, nchar(seq_str)))
            }
        }
    })))
}

base::lapply(c("n", "c"), function(nc) {
    base::lapply(
        c(1, 2, 3, 4, 6), function(ty) {
            p_fastadb <- get_fasta_db(ty, "p", nc)
            n_fastadb <- get_fasta_db(ty, "n", nc)

            fig <- ggseqlogo::ggseqlogo(list(
                "P" = p_fastadb,
                "N" = n_fastadb
            ), ncol = 2)

            ggplot2::ggsave(
                stringr::str_interp(
                    "tmp/data_out_md_docs/research/\
                    T${ty}/T${ty}_${nc}ter_weblogo.pdf"
                ),
                fig,
                width = 40,
                height = 10
            )
        }
    )
})
get_scratch_db <- function(ty, pn, tagstyle) {
    return(unlist(base::lapply(as.character(Biostrings::readBStringSet(
        stringr::str_interp(
            "tmp/scratchfasta/T${ty}/${tagstyle}/${pn}.fasta",
        ),
        format = "fasta",
        nrec = -1L,
        skip = 0L,
        seek.first.rec = FALSE,
        use.names = FALSE
    )), function(seq_str) {
        if (nchar(seq_str) <= 100) {
            return(paste(
                c(
                    seq_str,
                    paste(
                        replicate(100 - nchar(seq_str), "B"),
                        collapse = ""
                    )
                ),
                collapse = ""
            ))
        } else {
            return(substr(seq_str, 1, 100))
        }
    })))
}

base::lapply(c("acc", "ssa"), function(tagstyle) {
    base::lapply(
        c(1, 2, 3, 4, 6), function(ty) {
            p_fastadb <- get_scratch_db(ty, "p", tagstyle)
            n_fastadb <- get_scratch_db(ty, "n", tagstyle)

            fig <- ggseqlogo::ggseqlogo(list(
                "P" = p_fastadb,
                "N" = n_fastadb
            ), ncol = 2)

            ggplot2::ggsave(
                stringr::str_interp(
                    paste(
                        c(
                            "tmp/data_out_md_docs/research/",
                            "T${ty}/T${ty}_${tagstyle}_weblogo.pdf"
                        ),
                        collapse = ""
                    )
                ),
                fig,
                width = 40,
                height = 10
            )
        }
    )
})