"""Offline preflight; NO claims about LLM learning."""
from .run import simulate


def main():
    seeds = (1319, 1320, 1321, 1322, 1323)
    for family in ('additive', 'interaction'):
        print('\nHidden family:', family)
        for policy in ('memorize', 'additive'):
            runs = [simulate(policy, 'none', family, seed, 7, False, verbose=False)['posttest']['accuracy'] for seed in seeds]
            print(f'{policy:10s} held-out avg {sum(runs)/len(runs):.1%} (n={len(seeds)})')


if __name__ == '__main__':
    main()
