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
WHERE ci.note = '(as Barbara Demuth-Honauer)'
  AND cn.country_code = '[im]'
  AND it.info = 'crazy credits'
  AND mc.note LIKE '%DVD%'
  AND (mc.note LIKE '%TV)%'
       OR mc.note LIKE '%A) %')
  AND mi.info IS NOT NULL
  AND (mi.info LIKE '%ore%'
       OR mi.info LIKE '%4 J%')
  AND n.gender = 'f'
  AND n.name LIKE '%on,%'
  AND rt.role = 'writer'
  AND t.production_year BETWEEN 2007 AND 2008
  AND t.title LIKE '%xCo%'
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

