for F in 0 1;
  do
    for a in 2 3
      do
        for k in `seq 1 6`
          do
            #if [ ! -f results_"${F}"_"${a}"_"${k}".txt ]; then
              ./pnis_guarding.py "-a" "${a}" "-f" "$F" "--step" "${k}"  > results_"${F}"_"${a}"_"${k}".txt
            #fi
          done
      done
  done
