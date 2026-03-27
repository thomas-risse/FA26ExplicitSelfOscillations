#ifndef ARTICULATION_H
#define ARTICULATION_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include "Vowels.h"

class Articulation
{
private:
    std::vector<float> areas, positions, etas;

public:
    Articulation(vowels::Vowel const &v = vowels::a)
    {
        setFromVowel(v);
    };

    void setFromVowel(vowels::Vowel const &v)
    {
        positions = v.positions;
        areas = v.areas;
        etas = v.etas;
    };

    void interpolate2Vowels(vowels::Vowel const &v1, vowels::Vowel const &v2, float alpha)
    {
        for (size_t i = 0; i < positions.size(); ++i)
        {
            positions[i] = v1.positions[i] + alpha * (v2.positions[i] - v1.positions[i]);
            areas[i] = v1.areas[i] + alpha * (v2.areas[i] - v1.areas[i]);
            etas[i] = v1.etas[i] + alpha * (v2.etas[i] - v1.etas[i]);
        }
    };

    void interpolateNVowels(vowels::Vowel const *vs, float const *alphas, size_t const n)
    {
        float sumAlpha = 0;
        for (size_t i = 0; i < n; ++i)
        {
            sumAlpha += alphas[i];
        }
        for (size_t i = 0; i < positions.size(); ++i)
        {
            positions[i] = 0;
            areas[i] = 0;
            etas[i] = 0;
            for (size_t j = 0; j < n; ++j)
            {
                positions[i] += alphas[j] / sumAlpha * vs[j].positions[i];
                areas[i] += alphas[j] / sumAlpha * vs[j].areas[i];
                etas[i] += alphas[j] / sumAlpha * vs[j].etas[i];
            }
        }
    };

    // Find N closest vowels to target F0 and F1 frequencies and compute interpolation coefficients
    void findClosestVowelsForFormants(float targetF0, float targetF1, size_t N,
                                      std::vector<vowels::Vowel> &closestVowels,
                                      std::vector<float> &alphas)
    {
        if (N == 0)
            return;

        // Create vector of pairs: (distance, vowel pointer)
        std::vector<std::pair<float, vowels::Vowel const *>> vowelDistances;

        // Calculate distance for each vowel
        for (const auto &vowel : vowels::vowels)
        {
            if (vowel.Ftheos.size() >= 2)
            {
                float f0 = vowel.Ftheos[0];
                float f1 = vowel.Ftheos[1];

                // Euclidean distance in F0-F1 space
                float distance = std::sqrt((targetF0 - f0) * (targetF0 - f0) +
                                           (targetF1 - f1) * (targetF1 - f1));

                vowelDistances.push_back({distance, &vowel});
            }
        }

        // Sort by distance (closest first)
        std::sort(vowelDistances.begin(), vowelDistances.end());

        // Take the N closest vowels
        size_t numVowels = std::min(N, vowelDistances.size());
        closestVowels.clear();
        closestVowels.reserve(numVowels);

        // Calculate weights based on inverse distance
        std::vector<float> weights;
        weights.reserve(numVowels);

        for (size_t i = 0; i < numVowels; ++i)
        {
            closestVowels.push_back(*vowelDistances[i].second); // Dereference to get the vowel object

            // Use inverse distance weighting (with small epsilon to avoid division by zero)
            float distance = vowelDistances[i].first;
            float weight = 1.0f / (distance + 1e-6f);
            weights.push_back(weight);
        }

        // Normalize weights to sum to 1
        float totalWeight = 0.0f;
        for (float w : weights)
        {
            totalWeight += w;
        }

        alphas.clear();
        alphas.reserve(numVowels);
        for (float w : weights)
        {
            alphas.push_back(w / totalWeight);
        }
    }

    // Convenience function that directly sets the articulation from F0/F1
    void setFromFormants(float targetF0, float targetF1, size_t N = 3)
    {
        std::vector<vowels::Vowel> closestVowels;
        std::vector<float> alphas;

        findClosestVowelsForFormants(targetF0, targetF1, N, closestVowels, alphas);

        if (!closestVowels.empty())
        {
            interpolateNVowels(closestVowels.data(), alphas.data(), closestVowels.size());
        }
    }

    template <typename T>
    void getAreas(T const *evalPositions, T *out, std::size_t const size)
    {
        double xscale = positions[5] / evalPositions[size - 1];
        for (size_t i = 0; i < size; ++i)
        {
            if (evalPositions[i] * xscale < positions[0])
            {
                out[i] = areas[0];
            }
            else if (evalPositions[i] * xscale < positions[1])
            {
                out[i] = (areas[1] + areas[0]) / 2 + (areas[1] - areas[0]) / 2 * std::cos(M_PI * std::pow((positions[1] - evalPositions[i] * xscale) / (positions[1] - positions[0]), etas[0]));
            }
            else if (evalPositions[i] * xscale < positions[2])
            {
                out[i] = (areas[2] + areas[1]) / 2 + (areas[2] - areas[1]) / 2 * std::cos(M_PI * std::pow((positions[2] - evalPositions[i] * xscale) / (positions[2] - positions[1]), etas[1]));
            }
            else if (evalPositions[i] * xscale < positions[3])
            {
                out[i] = (areas[3] + areas[2]) / 2 + (areas[3] - areas[2]) / 2 * std::cos(M_PI * std::pow((positions[3] - evalPositions[i] * xscale) / (positions[3] - positions[2]), etas[2]));
            }
            else if (evalPositions[i] * xscale < positions[4])
            {
                out[i] = (areas[4] + areas[3]) / 2 + (areas[4] - areas[3]) / 2 * std::cos(M_PI * std::pow((positions[4] - evalPositions[i] * xscale) / (positions[4] - positions[3]), etas[3]));
            }
            else
            {
                out[i] = areas[5];
            }
            // Bound the area
            out[i] = std::max(T(1e-8), out[i]);
        }
    };
};
#endif
