﻿# spider-5-in-a-row
1.py文件说明：
MSDM5002FINALSTRATEGYbigBOSS.py是Player vs AI脚本（人机对战）
MSDM5002FINALBATTLE.py是AI vs AI脚本（AI对战）
MSDM5002FINALSAMPLE.py是AI vs AI中用于对战的傻瓜AI
2.AI vs AI脚本使用说明
1）将module_path改为本文件夹所在位置
2）与另外团队进行对战时，只需将“import MSDM5002FINALSAMPLE as u”里的MSDM5002FINALSAMPLE改为对面团队的.py脚本即可
3）下棋顺序可在运行主程序修改，选择“battle(gp1.ai_move, u.computer_move)”或“battle(u.computer_move, gp1.ai_move)”
3.注意事项
1）MSDM5002FINALSTRATEGYbigBOSS.py用于检测胜利的check_winner函数放在了MCTS类中
2）用于AI对战的MSDM5002FINALBATTLE.py中已将check_winner函数放在py中可以直接调用，不需要再使用"gp1.check_winner"或"gp2.check_winner",直接使用"check_winner(board)"即可
