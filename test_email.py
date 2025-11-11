import smtplib
from email.message import EmailMessage
import ssl

# --- 1. 配置你的邮件信息 ---

# SMTP 服务器地址和端口 (这里以 Gmail 为例)
# - Gmail: ('smtp.gmail.com', 587)
# - 163邮箱: ('smtp.163.com', 587 或 465)
# - QQ邮箱: ('smtp.qq.com', 587 或 465)
SMTP_SERVER = 'smtp.163.com'
SMTP_PORT = 465  # 对于 STARTTLS

SENDER_EMAIL = '17799138830@163.com'       # 你的邮箱地址
SENDER_PASSWORD = 'SAcjU37U7dH5VsXA'  # 你的 "应用专用密码"
RECEIVER_EMAIL = 'yangjunx21@gmail.com' # 收件人邮箱地址

# ----------------------------


# --- 2. 创建邮件对象 ---
msg = EmailMessage()
msg['Subject'] = '【163 邮箱 Port 465】测试邮件' # 邮件主题
msg['From'] = SENDER_EMAIL                   # 发件人
msg['To'] = RECEIVER_EMAIL                     # 收件人
msg.set_content('你好，\n\n这是使用 163 邮箱 465 端口发送的邮件正文。\n\n祝好！') # 邮件正文

print("正在创建邮件...")

# --- 3. 连接并发送邮件 (使用 SMTP_SSL) ---
try:
    # 创建一个默认的 SSL 上下文
    context = ssl.create_default_context()
    
    print(f"正在通过 SSL 连接到 {SMTP_SERVER}:{SMTP_PORT}...")
    
    # 使用 smtplib.SMTP_SSL()，它会立即建立 SSL 连接
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        
        # 登录 SMTP 服务器
        # 再次确认: SENDER_PASSWORD 必须是 163 邮箱设置中生成的 "授权码"
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        print("登录成功！")
        
        # 发送邮件
        server.send_message(msg)
        print(f"邮件已成功发送至 {RECEIVER_EMAIL}！")
        
        # (使用 'with' 语句，连接会自动关闭)

except smtplib.SMTPException as e:
    print(f"SMTP 错误: {e}")
except Exception as e:
    print(f"发送邮件时出错: {e}")