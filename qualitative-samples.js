// Camera-ready figure query identities, verified against the final paper PDF.
(() => {
  const frames = {
    'uav-in': 'DJI_20231207160946_0244_W.JPG',
    'uav-out': 'DJI_20231018101314_0007_D.JPG',
    'swiss-in': 'urbanscape-2020-09-24-planned_01378_100_0008_0203_img.jpg',
    'swiss-out': 'urbanscape-oop-2021-10-14-outofplace_01152_DJI_01152_img.jpg'
  };
  window.PLANARCITYLOC_RESULTS.samples.forEach(sample => {
    if (frames[sample.id]) Object.assign(sample, { frame: frames[sample.id], sourceLabel: 'Camera-ready paper figure' });
  });
})();

// Additional matched overlays re-rendered from archived poses; no new inference.
window.PLANARCITYLOC_RESULTS.samples.push(...[
  {
    "id": "uav-in-extra-01",
    "dataset": "uav",
    "split": "in-Traj",
    "frame": "DJI_20231207163021_0232_W.JPG",
    "width": 960,
    "height": 720,
    "errors": {
      "v2": [
        6.135772415996542,
        2.7408834349050197
      ],
      "v3": [
        5.402231447644625,
        2.740868647243102
      ],
      "ours": [
        0.7994582586645225,
        0.40131238240237044
      ]
    },
    "origin": "ambiguity",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived geometry-ambiguity experiment; rendered from its saved final pose."
  },
  {
    "id": "uav-in-extra-02",
    "dataset": "uav",
    "split": "in-Traj",
    "frame": "DJI_20231207152632_0162_W.JPG",
    "width": 960,
    "height": 720,
    "errors": {
      "v2": [
        2.2390893521777118,
        0.8199340490768758
      ],
      "v3": [
        1.1801568806278306,
        0.823723666647992
      ],
      "ours": [
        1.0550123637247597,
        0.6294298610913313
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-in-extra-03",
    "dataset": "uav",
    "split": "in-Traj",
    "frame": "DJI_20231207154028_0562_W.JPG",
    "width": 960,
    "height": 720,
    "errors": {
      "v2": [
        3.0718304088568513,
        1.6402712801004389
      ],
      "v3": [
        2.139485829886632,
        1.6169761080717242
      ],
      "ours": [
        7.336498063284041,
        2.982585455973581
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-in-extra-04",
    "dataset": "uav",
    "split": "in-Traj",
    "frame": "DJI_20231207160939_0241_W.JPG",
    "width": 960,
    "height": 720,
    "errors": {
      "v2": [
        6.906431586485122,
        2.5327055308693764
      ],
      "v3": [
        5.833535515449559,
        2.5226280823225196
      ],
      "ours": [
        0.924567118230607,
        0.17973251500632687
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-in-extra-05",
    "dataset": "uav",
    "split": "in-Traj",
    "frame": "DJI_20231207163317_0294_W.JPG",
    "width": 960,
    "height": 720,
    "errors": {
      "v2": [
        4.823186890811752,
        2.159320882102138
      ],
      "v3": [
        3.964074925098022,
        2.101366492072387
      ],
      "ours": [
        0.5531272362018115,
        0.3042994673656019
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-out-extra-01",
    "dataset": "uav",
    "split": "out-of-Traj",
    "frame": "DJI_20231018093439_0184_D.JPG",
    "width": 960,
    "height": 719,
    "errors": {
      "v2": [
        1.5263551800212256,
        0.7622961464283269
      ],
      "v3": [
        2.116333749791865,
        0.755606736891126
      ],
      "ours": [
        2.1525455645590874,
        0.7665302418709347
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-out-extra-02",
    "dataset": "uav",
    "split": "out-of-Traj",
    "frame": "DJI_20231018101032_0925_D.JPG",
    "width": 960,
    "height": 719,
    "errors": {
      "v2": [
        0.6342914404261976,
        0.8150267274004886
      ],
      "v3": [
        1.9548711889544774,
        0.8069769051834302
      ],
      "ours": [
        0.6524804127669598,
        0.6717846054967545
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-out-extra-03",
    "dataset": "uav",
    "split": "out-of-Traj",
    "frame": "DJI_20231018102151_0255_D.JPG",
    "width": 960,
    "height": 719,
    "errors": {
      "v2": [
        1.1445416663303052,
        0.758021052066339
      ],
      "v3": [
        1.8137610324555438,
        0.7471850315842818
      ],
      "ours": [
        0.4179042270833139,
        0.4534501184034057
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-out-extra-04",
    "dataset": "uav",
    "split": "out-of-Traj",
    "frame": "DJI_20231018114601_0627_D.JPG",
    "width": 960,
    "height": 719,
    "errors": {
      "v2": [
        2.4272336458537396,
        1.3438897081155057
      ],
      "v3": [
        4.057522083982588,
        1.3541409039780918
      ],
      "ours": [
        2.1355032299014036,
        0.8499640426517284
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "uav-out-extra-05",
    "dataset": "uav",
    "split": "out-of-Traj",
    "frame": "DJI_20231018121801_0318_D.JPG",
    "width": 960,
    "height": 719,
    "errors": {
      "v2": [
        4.493427224217922,
        1.4773008496367652
      ],
      "v3": [
        1.4901556117420511,
        1.4420145777632056
      ],
      "ours": [
        1.1084167522878487,
        0.633680679262055
      ]
    },
    "origin": "saved-eval",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived MoGe-2 ViT-S / SAM 3 evaluation (stride 10); rendered from its saved final pose."
  },
  {
    "id": "swiss-in-extra-01",
    "dataset": "swiss",
    "split": "in-Place",
    "frame": "urbanscape-2020-09-17-planned_00622_100_0039_0390_img.jpg",
    "width": 720,
    "height": 480,
    "errors": {
      "v2": [
        9.176309118566575,
        3.5962367281490635
      ],
      "v3": [
        2.40002124951145,
        0.994404560198712
      ],
      "ours": [
        2.8837924094110026,
        1.8056306100729982
      ]
    },
    "origin": "ambiguity",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived geometry-ambiguity experiment; rendered from its saved final pose."
  },
  {
    "id": "swiss-out-extra-01",
    "dataset": "swiss",
    "split": "out-of-Place",
    "frame": "urbanscape-oop-2021-10-14-outofplace_00353_DJI_00353_img.jpg",
    "width": 720,
    "height": 480,
    "errors": {
      "v2": [
        4.813699553819412,
        1.2438578095524209
      ],
      "v3": [
        1.001649195522704,
        0.2763635936599906
      ],
      "ours": [
        4.967131709166501,
        1.615955348686867
      ]
    },
    "origin": "ambiguity",
    "sourceLabel": "Additional saved-run example",
    "sourceNote": "Archived geometry-ambiguity experiment; rendered from its saved final pose."
  }
]);
