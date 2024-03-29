# universal property experiments
for m in 0;
  do
    for F in 1;
      do
        for a in 2 3
          do
            for k in `seq 1 6`
              do
                if [ ! -f results_m"${m}"_f"${F}"_a"${a}"_k"${k}".txt ]; then
                  ./pnis_guarding.py "-m" "${m}" "-a" "${a}" "-f" "$F" "--step" "${k}"  > results_m"${m}"_f"${F}"_a"${a}"_k"${k}".txt
                fi
              done
          done
      done
  done

# existential property experiments
for F in 0;
  do
    for a in 2
      do
        for k in `seq 2 5`
          do
            for th in `seq 2 6`
              do
                if [ ! -f results_f"${F}"_k"${k}"_"${a}"_"${th}".txt ]; then
                  ./pnis_guarding.py "-a" "${a}" "-n" "${th}" "-f" "$F" "--step" "${k}"  > results_f"${F}"_k"${k}"_"${a}"_"${th}".txt
                fi
              done
          done
      done
  done
