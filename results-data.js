// Full-split evaluation results and qualitative examples.
window.PLANARCITYLOC_RESULTS = {
  "source": {
    "source": "PlanarCityLoc full-split evaluation",
    "metrics": [
      "Recall at 2m–2° (%)",
      "Recall at 3m–3° (%)",
      "Recall at 5m–5° (%)",
      "Median translation error (m)",
      "Median rotation error (°)"
    ],
    "note": "Recall at each threshold requires both translation and rotation errors to be below the threshold."
  },
  "datasets": [
    {
      "id": "uav",
      "name": "UAVD4L-LoDv2",
      "lod": "LoD3",
      "splits": [
        {
          "name": "in-Traj",
          "index": 0,
          "sample": "uav-in"
        },
        {
          "name": "out-of-Traj",
          "index": 1,
          "sample": "uav-out"
        }
      ]
    },
    {
      "id": "swiss",
      "name": "Swiss-EPFLv2",
      "lod": "LoD2",
      "splits": [
        {
          "name": "in-Place",
          "index": 2,
          "sample": "swiss-in"
        },
        {
          "name": "out-of-Place",
          "index": 3,
          "sample": "swiss-out"
        }
      ]
    }
  ],
  "rows": [
    {
      "id": 0,
      "method": "Released priors",
      "map": "Sensor prior",
      "model": "—",
      "setup": "—",
      "ours": false,
      "rankingGroup": null,
      "metrics": [
        [
          0,
          0,
          4.3,
          6.48,
          1.63
        ],
        [
          0,
          0,
          0.4,
          11.12,
          1.33
        ],
        [
          0,
          0,
          0.6,
          17.59,
          3.86
        ],
        [
          0,
          0,
          1.1,
          17.88,
          3.94
        ]
      ]
    },
    {
      "id": 1,
      "method": "UAVLoc",
      "map": "Textured mesh",
      "model": "LoFTR",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "textured",
      "metrics": [
        [
          84.98,
          88.09,
          88.9,
          0.81,
          0.29
        ],
        [
          91.56,
          92.02,
          92.11,
          0.79,
          0.2
        ],
        [
          26.4,
          46.21,
          62.22,
          3.06,
          1.86
        ],
        [
          68.87,
          81,
          84.96,
          1.12,
          1.08
        ]
      ]
    },
    {
      "id": 2,
      "method": "UAVLoc",
      "map": "Textured mesh",
      "model": "RoMa",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "textured",
      "metrics": [
        [
          93.27,
          93.7,
          93.77,
          0.78,
          0.25
        ],
        [
          95.03,
          95.53,
          95.53,
          0.73,
          0.22
        ],
        [
          45.95,
          66.77,
          80.73,
          2.08,
          1.29
        ],
        [
          89.18,
          89.68,
          98.84,
          0.77,
          1.04
        ]
      ]
    },
    {
      "id": 3,
      "method": "LoD-Loc",
      "map": "LoD model",
      "model": "Custom U-Net",
      "setup": "In-domain training",
      "ours": false,
      "rankingGroup": "trained",
      "metrics": [
        [
          49.56,
          71.82,
          89.09,
          3.32,
          1.48
        ],
        [
          54.2,
          75.05,
          89.51,
          3.33,
          1.18
        ],
        [
          36.79,
          50.56,
          69.77,
          2.87,
          1.78
        ],
        [
          14.24,
          31.39,
          59.89,
          8.73,
          2.78
        ]
      ]
    },
    {
      "id": 4,
      "method": "LoD-Loc v2",
      "map": "LoD model",
      "model": "SAM2 + U-Net",
      "setup": "In-domain training",
      "ours": false,
      "rankingGroup": "trained",
      "metrics": [
        [
          93.7,
          98.4,
          99.5,
          0.72,
          0.15
        ],
        [
          97.9,
          99.8,
          100,
          0.71,
          0.14
        ],
        [
          54.2,
          74.6,
          92,
          1.83,
          0.85
        ],
        [
          31.4,
          58.53,
          86.3,
          2.64,
          0.73
        ]
      ]
    },
    {
      "id": 5,
      "method": "LoD-Loc v3",
      "map": "LoD model",
      "model": "SAM-Prompter",
      "setup": "Cross-dataset*",
      "ours": false,
      "rankingGroup": "trained",
      "metrics": [
        [
          97.6,
          98.9,
          99.7,
          0.49,
          0.13
        ],
        [
          97.4,
          99,
          99.4,
          0.6,
          0.12
        ],
        [
          58.6,
          79.9,
          95.4,
          1.61,
          0.77
        ],
        [
          36.9,
          64.1,
          88.9,
          2.43,
          0.83
        ]
      ]
    },
    {
      "id": 6,
      "method": "LoD-Loc v2 †",
      "map": "LoD model",
      "model": "—",
      "setup": "Oracle masks",
      "ours": false,
      "rankingGroup": "oracle",
      "metrics": [
        [
          57.2,
          78.8,
          92.8,
          1.85,
          1.23
        ],
        [
          64.9,
          94.5,
          98.3,
          1.76,
          0.94
        ],
        [
          74.9,
          90.6,
          93.5,
          1.36,
          0.85
        ],
        [
          43.3,
          75.2,
          93.9,
          2.15,
          0.78
        ]
      ]
    },
    {
      "id": 7,
      "method": "LoD-Loc v3 †",
      "map": "LoD model",
      "model": "—",
      "setup": "Oracle masks",
      "ours": false,
      "rankingGroup": "oracle",
      "metrics": [
        [
          54.5,
          78.7,
          90.6,
          1.87,
          1.23
        ],
        [
          41.3,
          83.8,
          96.4,
          2.15,
          0.95
        ],
        [
          56.3,
          72.2,
          78.9,
          1.79,
          0.91
        ],
        [
          40.1,
          73.6,
          93.9,
          2.27,
          0.79
        ]
      ]
    },
    {
      "id": 8,
      "method": "MC-Loc",
      "map": "LoD model",
      "model": "DINOv2",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          1.2,
          4.1,
          17.4,
          8.29,
          2.58
        ],
        [
          2.4,
          7.4,
          26.1,
          7.02,
          2.29
        ],
        [
          0.9,
          4.4,
          17.5,
          8.18,
          2.54
        ],
        [
          2.9,
          9,
          30.2,
          6.23,
          1.97
        ]
      ]
    },
    {
      "id": 9,
      "method": "MC-Loc",
      "map": "LoD model",
      "model": "RoMa",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          0.1,
          0.6,
          3.3,
          10.6,
          8.6
        ],
        [
          0.2,
          0.9,
          3.3,
          16.9,
          3.88
        ],
        [
          0.2,
          1.2,
          4.8,
          9.8,
          2.65
        ],
        [
          0.7,
          2.1,
          11.5,
          10.3,
          3.97
        ]
      ]
    },
    {
      "id": 10,
      "method": "LoD-Loc v2 †",
      "map": "LoD model",
      "model": "SAM3",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          35.7,
          67.8,
          81.5,
          2.33,
          1.27
        ],
        [
          64.3,
          93.3,
          98.2,
          1.77,
          0.95
        ],
        [
          9.4,
          19.9,
          39.7,
          6.51,
          1.83
        ],
        [
          17.4,
          33,
          66.2,
          3.9,
          1.33
        ]
      ]
    },
    {
      "id": 11,
      "method": "PlanarCityLoc · cost volume",
      "map": "LoD model · initialization",
      "model": "MoGe-2 + SAM3",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          27.6,
          59.9,
          79.4,
          2.62,
          1.49
        ],
        [
          34.9,
          74.1,
          95.3,
          2.35,
          1.22
        ],
        [
          15.4,
          30.2,
          48.7,
          5.04,
          1.72
        ],
        [
          11.3,
          23.2,
          57.5,
          4.42,
          1.41
        ]
      ]
    },
    {
      "id": 12,
      "method": "PlanarCityLoc · + particle filter",
      "map": "LoD model · coarse pose",
      "model": "MoGe-2 + SAM3",
      "setup": "Off-the-shelf",
      "ours": false,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          36,
          67.7,
          82.8,
          2.36,
          1.3
        ],
        [
          60.8,
          90.6,
          98.5,
          1.81,
          0.98
        ],
        [
          24.9,
          41.6,
          61.8,
          3.54,
          1.54
        ],
        [
          14.5,
          31.1,
          70.2,
          3.99,
          1.23
        ]
      ]
    },
    {
      "id": 13,
      "method": "PlanarCityLoc (ours)",
      "map": "LoD model · + plane-based PnP",
      "model": "MoGe-2 + SAM3",
      "setup": "Off-the-shelf",
      "ours": true,
      "rankingGroup": "off-the-shelf",
      "metrics": [
        [
          75.4,
          90.8,
          97.2,
          1.28,
          0.68
        ],
        [
          80.6,
          95.1,
          99,
          1.26,
          0.69
        ],
        [
          20.6,
          39,
          59.7,
          3.7,
          1.9
        ],
        [
          48.5,
          72.3,
          89.4,
          1.98,
          1.21
        ]
      ]
    }
  ],
  "methods": [
    {
      "id": "v2",
      "name": "LoD-Loc v2",
      "setting": "SAM 3",
      "label": "LoD-Loc v2 · SAM 3"
    },
    {
      "id": "particle",
      "name": "Particle filter",
      "setting": "PlanarCityLoc coarse pose",
      "label": "Particle filter"
    },
    {
      "id": "v3",
      "name": "LoD-Loc v3",
      "setting": "oracle masks",
      "label": "LoD-Loc v3 · oracle masks"
    },
    {
      "id": "ours",
      "name": "PlanarCityLoc",
      "setting": "MoGe-2 + SAM 3",
      "label": "PlanarCityLoc (ours)"
    },
    {
      "id": "gt",
      "name": "Ground truth",
      "setting": "GT pose",
      "label": "Ground truth"
    }
  ],
  "samples": [
    {
      "id": "uav-in",
      "dataset": "uav",
      "split": "in-Traj",
      "width": 425,
      "height": 319,
      "methods": ["gt", "v2", "particle", "ours"],
      "errors": {
        "v2": [
          7.67,
          2.55
        ],
        "particle": [
          6.79,
          2.56
        ],
        "ours": [
          0.68,
          0.32
        ]
      }
    },
    {
      "id": "uav-out",
      "dataset": "uav",
      "split": "out-of-Traj",
      "width": 425,
      "height": 319,
      "methods": ["gt", "v2", "particle", "ours"],
      "errors": {
        "v2": [
          2.38,
          1.03
        ],
        "particle": [
          2.24,
          1.03
        ],
        "ours": [
          1.01,
          0.54
        ]
      }
    },
    {
      "id": "swiss-in",
      "dataset": "swiss",
      "split": "in-Place",
      "width": 425,
      "height": 319,
      "methods": ["gt", "v2", "particle", "ours"],
      "errors": {
        "v2": [
          6.50,
          2.56
        ],
        "particle": [
          2.23,
          1.22
        ],
        "ours": [
          2.99,
          2.93
        ]
      }
    },
    {
      "id": "swiss-out",
      "dataset": "swiss",
      "split": "out-of-Place",
      "width": 425,
      "height": 319,
      "methods": ["gt", "v2", "particle", "ours"],
      "errors": {
        "v2": [
          3.93,
          1.22
        ],
        "particle": [
          5.09,
          1.23
        ],
        "ours": [
          1.03,
          0.5
        ]
      }
    }
  ]
};
