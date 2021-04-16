---
title: Tạo blog với vuepress
image: https://picsum.photos/536/354/?random&date=2019-02-20
date: 2021-04-17
display: home
tags: 
  - frontend
  - vuepress
categories:
  - Vue
--- 
# Mục lục
[[toc]]

## Tạo project vue press:
Tại giao diện terminal, bạn gõ lệnh để tải và cài đặt vuepress
```
npx create-vuepress-site blog4share

```
bạn sẽ được hỏi một số thông tin về project, để đơn giản cứ enter hết là xong.

Kế tiếp bạn vào thư mục blog4share/docs
```
cd blog4share/docs
```
Tại đây, để chạy trang vuepress bạn gõ lệnh
```
npm run dev
```
Khi xuất hiện dòng chữ bên dưới tức là đã cài được vuepress thành công và http://localhost:8080/ cũng là địa chỉ để truy cập vào vuepress.
```
> VuePress dev server listening at http://localhost:8080/
```
khi vào địa chỉ phía trên sẽ ra giao diện như hình bên dưới
![](https://bl6pap004files.storage.live.com/y4mnhqv0n3WbKi3tX7KV8N54EPPGsLU8WrfDVwVYVYaUQDJAqx4INHbSSPATBQFGialtk9K3SYTsWWyyiosMYQ1oJNwvrbTZu7aptRs2041RvtU0mVnVkOlTVrHNV0MBvWtWX6Z2jVGDBLWeF_AE-lrXPzAqCQ2-RAbom8X9znifwumr_d5H7hAxkHFxgK3BFzo?width=1366&height=657&cropmode=none)

Ở bước tiếp theo chúng ta sẽ cài plugin blog cho vuepress.

## Cài theme blog cho vue press
Để cài theme blog, bạn chạy lệnh
```
npm install @vuepress/theme-blog -D
```
Sau khi chạy xong, bạn sửa file src/.vuepress/config.js với nội dung như dưới đây
```js
// src/.vuepress/config.js
module.exports = {
  title: 'Blog 4 Sharing', // Title for the site. This will be displayed in the navbar.
  theme: '@vuepress/theme-blog',
  themeConfig: {
    nav: [
      {
        text: "Trang chủ",
        link: "/",
      },
      {
        text: "Bài viết",
        link: "/posts/",
      },
    ],   
  }
}
```
Chạy đoạn lệnh bên dưới để kiểm tra
```
npm run dev
```
Nếu kết quả như hình dưới tức là đã cài xong theme blog.
![](https://bl6pap004files.storage.live.com/y4mUkRCL0tut_LvEll0Ti56saL3d01HERbymzZfnlL8ANZuxRIlzHSASc5Ah75iazSViMS3MgPpWkwimP1B4vUQaKHAUV1OS6QWNDS-kzl2QNA4l1iHa6D40aEKvQm4SCirraPXsWwjRdsvuKJmSqVx7QghcIIUGMTBCCQ84Jex_t2Dqpt6_LngrYFwh553B_JL?width=1366&height=657&cropmode=none)

Nhưng rồi cũng đâu giống trang blog nhỉ, danh sách các bài viết ở đâu?

Cái đó thì cần viết component qua bước kế tiếp để thực hiện điều đó nhé.

## Viết components để hiển thị bài viết
Vào thư mục src/.vuepress/components tạo một file có tên BlogIndex.vue với nội dung như bên dưới
```html
<!--src/.vuepress/components/BlogIndex.vue -->

<template>
<div>
    <div v-for="post in posts">
        <h2>
            <router-link :to="post.path">{{ post.frontmatter.title }}</router-link>
        </h2>
        
        <p>{{ post.frontmatter.description }}</p>

        <p><router-link :to="post.path">Read more</router-link></p>
    </div>
</div>
</template>

<script>
export default {
    computed: {
        posts() {
            return this.$site.pages
                .filter(x => x.path.startsWith('/posts/') && !x.frontmatter.blog_index)
                .sort((a, b) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date));
        }
    }
}
</script>
```
Đoạn script này đại ý là lấy toàn bộ những url có chứa /posts/ vào danh sách hiển thị.

Bạn tạo thư mục tại src/posts và tạo các bài viết theo mẫu file bên dưới
```
<!-- /posts/first-post.md --> không copy đòng này
# Using Vue in Markdown 3

## Browser API Access Restrictions

Because VuePress applications are server-rendered in Node.js when generating static builds, any Vue usage must conform to the [universal code requirements](https://ssr.vuejs.org/en/universal.html). In short, make sure to only access Browser / DOM APIs in `beforeMount` or `mounted` hooks.

If you are using or demoing components that are not SSR friendly (for example containing custom directives), you can wrap them inside the built-in `<ClientOnly>` component:

##

```
cũng tại thư mục posts, bạn tạo file index.md có nội dung như sau
```
<!-- /posts/index.md --> không copy đòng này
---
blog_index: true
---

# Blog

Welcome on Blog 4 Share

<BlogIndex />
```
Chạy đoạn lệnh bên dưới để kiểm tra
```
npm run dev
```
![](https://bl6pap004files.storage.live.com/y4mydY4H6_a_UgNhv4RG__oW_2XGbgynTmtxIKwO8LffbM1Hd7u599SUhH8GAaxGIQFu39h-WZGUztCxgWzjd9TO0rBcSXXKt6iSrtuUlUzQvV5ebY9zjPjCQAnYBCsZScO9kswJtayDUzG32ty-LKS09y4gcQGXNUSB7sDyGnVMaNT-c8IPNCFCukX_B_UjY9j?width=1366&height=600&cropmode=none)
## Một số chỉnh sửa nho nhỏ cho trang blog 
Để chỉnh trang chủ, bạn vào file src/index.md đổi nội dung như bên dưới:
```
---
home: true
---

![](https://pixabay.com/get/g0470b665bcad720b02a56ec715b31d99e43cb743c4aefb08e820ca7bbcac52b67ae06246772302915577dbbe533d71d73c289e870b65e0d4a682f8b153c1b92d_1920.jpg)
<OtherComponent/>
<Bar/>
<demo-component/>
```
Để cấu hình thanh nav ở trên, bạn edit file src/.vuepress/config.js như bên dưới để tùy chỉnh
```js
// src/.vuepress/config.js

module.exports = {
    title: 'Blog 4 Share',
    themeConfig: {
        nav: [
            { text: 'Home', link: '/' },
            { text: 'Blog', link: '/blog/' }
        ]
    }
}
```
Cuối cùng là chạy:
```
npm run dev
```
Trang chủ hiện như hình bên dưới là thành công
![](https://bl6pap004files.storage.live.com/y4m3X6yMSl6M4jIL5DDnJNwrguL8XbG_CU4BoelcbnB1KlzUzWUjqi4W2k_HofpnK00fLUAJwuYDvOGHNBlvexQkOwJs5WhiH5rvV-Q0_P_YkAZoXezNdhMccyj-8PhvQ7N7v3LMxsgU3okoYHIKA6rLr0XeHk5_7Uf2tpsg4Yya__OBpDY2WamssR40MdnG54q?width=1366&height=657&cropmode=none)
Bài viết này chỉ tập trung những thao tác cốt lõi nhất để tạo một blog cơ bản với vuepress. Để tìm hiểu sâu hơn, bạn nên xem tài liệu hỗ trợ tại https://vuepress.vuejs.org/ .