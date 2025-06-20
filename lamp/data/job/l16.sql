SELECT MIN(n.name) AS voicing_actress,
       MIN(t.title) AS kung_fu_panda
FROM aka_name AS an,
     char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     info_type AS it,
     movie_companies AS mc,
     movie_info AS mi,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note = '(as Ramón Colominas)'
  AND cn.country_code = '[mu]'
  AND it.info = 'taglines'
  AND mc.note LIKE '%LC,%'
  AND (mc.note LIKE '%Pan%'
       OR mc.note LIKE '%ng %')
  AND mi.info IS NOT NULL
  AND (mi.info LIKE '%25 %'
       OR mi.info LIKE '%emb%')
  AND n.gender = 'f'
  AND n.name LIKE '%n, %'
  AND rt.role = 'director'
  AND t.production_year BETWEEN 2007 AND 2008
  AND t.title LIKE '%3.1%'
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND mc.movie_id = mi.movie_id
  AND mi.movie_id = ci.movie_id
  AND cn.id = mc.company_id
  AND it.id = mi.info_type_id
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND n.id = an.person_id
  AND ci.person_id = an.person_id
  AND chn.id = ci.person_role_id;

